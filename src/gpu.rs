//! GPU mining backend via OpenCL — **Multi-GPU** support.
//!
//! Enumerates ALL available OpenCL devices and runs them in parallel.
//! Each GPU gets its own queue and dispatches batches independently.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use alloy::primitives::{keccak256, B256, U256};
use eyre::{eyre, Result};
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

const KERNEL_SRC: &str = include_str!("keccak_kernel.cl");
const DEFAULT_BATCH: usize = 1 << 22; // 4,194,304 nonces/dispatch

/// A single GPU device context ready to mine.
struct GpuDevice {
    queue: Queue,
    program: Program,
    device_name: String,
    batch_size: usize,
}

/// Multi-GPU miner — holds all detected GPU devices.
pub struct GpuMiner {
    devices: Vec<GpuDevice>,
    batch_size: usize,
}

impl GpuMiner {
    /// Initialize OpenCL GPUs.
    /// - `gpu_indices = None` → use ALL GPUs
    /// - `gpu_indices = Some(vec![0])` → use only GPU 0
    /// - `gpu_indices = Some(vec![0, 1])` → use GPU 0 and 1
    pub fn new(batch_size: Option<usize>, gpu_indices: Option<Vec<usize>>) -> Result<Self> {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH);

        let platform = Platform::default();
        let all_devices = Device::list_all(platform)
            .map_err(|e| eyre!("failed to list OpenCL devices: {e}"))?;

        if all_devices.is_empty() {
            return Err(eyre!("no OpenCL devices found"));
        }

        let selected_devices: Vec<Device> = if let Some(indices) = gpu_indices {
            let mut devs = Vec::new();
            for idx in &indices {
                if *idx >= all_devices.len() {
                    return Err(eyre!(
                        "GPU index {} but only {} devices available",
                        idx,
                        all_devices.len()
                    ));
                }
                devs.push(all_devices[*idx]);
            }
            devs
        } else {
            all_devices
        };

        let mut devices = Vec::new();
        for device in &selected_devices {
            let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());

            let context = Context::builder()
                .platform(platform)
                .devices(*device)
                .build()?;
            let queue = Queue::new(&context, *device, None)?;
            let program = Program::builder()
                .src(KERNEL_SRC)
                .devices(*device)
                .build(&context)?;

            devices.push(GpuDevice {
                queue,
                program,
                device_name,
                batch_size,
            });
        }

        Ok(Self { devices, batch_size })
    }

    pub fn device_names(&self) -> Vec<&str> {
        self.devices.iter().map(|d| d.device_name.as_str()).collect()
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Legacy compat
    pub fn device_name(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.device_name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Self-test ALL GPUs.
    pub fn self_test(&self) -> Result<()> {
        for (i, dev) in self.devices.iter().enumerate() {
            self_test_device(dev)
                .map_err(|e| eyre!("GPU {} ({}) self-test failed: {e}", i, dev.device_name))?;
        }
        Ok(())
    }

    /// Mine across ALL GPUs in parallel threads.
    /// Each GPU gets a different nonce range (offset by device index * batch_size * large_stride).
    pub fn mine(
        &self,
        challenge: B256,
        difficulty: U256,
        start_nonce: u64,
        stop_flag: Arc<AtomicBool>,
        attempts_counter: Arc<AtomicU64>,
    ) -> Result<Option<u64>> {
        let num_gpus = self.devices.len();
        if num_gpus == 0 {
            return Ok(None);
        }

        // Single GPU — no threading overhead
        if num_gpus == 1 {
            return mine_on_device(
                &self.devices[0],
                challenge,
                difficulty,
                start_nonce,
                stop_flag,
                attempts_counter,
            );
        }

        // Multi-GPU: each GPU gets a nonce range spaced far apart
        let result: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);
        let nonce_spacing: u64 = (self.batch_size as u64) * 1_000_000; // huge gap per GPU

        std::thread::scope(|s| {
            for (gpu_idx, dev) in self.devices.iter().enumerate() {
                let stop_flag = &stop_flag;
                let attempts_counter = &attempts_counter;
                let result = &result;
                let gpu_start = start_nonce.wrapping_add((gpu_idx as u64) * nonce_spacing);

                s.spawn(move || {
                    match mine_on_device(
                        dev,
                        challenge,
                        difficulty,
                        gpu_start,
                        Arc::clone(stop_flag),
                        Arc::clone(attempts_counter),
                    ) {
                        Ok(Some(nonce)) => {
                            let mut r = result.lock().unwrap();
                            if r.is_none() {
                                *r = Some(nonce);
                            }
                            stop_flag.store(true, Ordering::Relaxed);
                        }
                        Ok(None) => {}
                        Err(e) => {
                            eprintln!("❌ GPU {} error: {e}", gpu_idx);
                        }
                    }
                });
            }
        });

        Ok(result.into_inner().unwrap())
    }
}

/// Run mining loop on a single GPU device.
fn mine_on_device(
    dev: &GpuDevice,
    challenge: B256,
    difficulty: U256,
    start_nonce: u64,
    stop_flag: Arc<AtomicBool>,
    attempts_counter: Arc<AtomicU64>,
) -> Result<Option<u64>> {
    let cw = split_challenge_le(&challenge);
    let dw = split_difficulty_be(difficulty);

    let found_nonce = Buffer::<u64>::builder()
        .queue(dev.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .copy_host_slice(&[0u64])
        .build()?;
    let found_flag = Buffer::<i32>::builder()
        .queue(dev.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .copy_host_slice(&[0i32])
        .build()?;

    let mut nonce_base: u64 = start_nonce;
    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return Ok(None);
        }

        found_flag.write(&[0i32][..]).enq()?;
        found_nonce.write(&[0u64][..]).enq()?;

        let kernel = Kernel::builder()
            .program(&dev.program)
            .name("mine_keccak")
            .queue(dev.queue.clone())
            .global_work_size(dev.batch_size)
            .arg(cw[0]).arg(cw[1]).arg(cw[2]).arg(cw[3])
            .arg(dw[0]).arg(dw[1]).arg(dw[2]).arg(dw[3])
            .arg(nonce_base)
            .arg(&found_nonce)
            .arg(&found_flag)
            .build()?;

        unsafe { kernel.enq()?; }
        dev.queue.finish()?;

        attempts_counter.fetch_add(dev.batch_size as u64, Ordering::Relaxed);

        let mut flag = [0i32];
        found_flag.read(&mut flag[..]).enq()?;
        if flag[0] != 0 {
            let mut got = [0u64];
            found_nonce.read(&mut got[..]).enq()?;
            let nonce = got[0];

            // CPU-verify before returning
            let h = cpu_hash(&challenge, U256::from(nonce));
            if U256::from_be_bytes::<32>(h.0) < difficulty {
                return Ok(Some(nonce));
            } else {
                eprintln!(
                    "⚠️  GPU reported nonce {} but CPU verify failed — skipping batch",
                    nonce
                );
            }
        }

        nonce_base = nonce_base.wrapping_add(dev.batch_size as u64);
        std::thread::sleep(Duration::from_micros(1));
    }
}

/// Self-test a single device.
fn self_test_device(dev: &GpuDevice) -> Result<()> {
    let challenge = B256::from(*b"abcdefghijklmnopqrstuvwxyz012345");
    let difficulty = U256::MAX;
    let nonce_base: u64 = 12345;

    let (cw, dw) = (split_challenge_le(&challenge), split_difficulty_be(difficulty));

    let found_nonce = Buffer::<u64>::builder()
        .queue(dev.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .copy_host_slice(&[0u64])
        .build()?;
    let found_flag = Buffer::<i32>::builder()
        .queue(dev.queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .copy_host_slice(&[0i32])
        .build()?;

    let kernel = Kernel::builder()
        .program(&dev.program)
        .name("mine_keccak")
        .queue(dev.queue.clone())
        .global_work_size(1usize)
        .arg(cw[0]).arg(cw[1]).arg(cw[2]).arg(cw[3])
        .arg(dw[0]).arg(dw[1]).arg(dw[2]).arg(dw[3])
        .arg(nonce_base)
        .arg(&found_nonce)
        .arg(&found_flag)
        .build()?;

    unsafe { kernel.enq()?; }
    dev.queue.finish()?;

    let mut flag = [0i32];
    found_flag.read(&mut flag[..]).enq()?;
    if flag[0] == 0 {
        return Err(eyre!("GPU did not report any hit against MAX difficulty"));
    }

    let mut got = [0u64];
    found_nonce.read(&mut got[..]).enq()?;
    if got[0] != nonce_base {
        return Err(eyre!(
            "GPU reported nonce {} but expected {}",
            got[0],
            nonce_base
        ));
    }

    let cpu_h = cpu_hash(&challenge, U256::from(nonce_base));
    if !(U256::from_be_bytes::<32>(cpu_h.0) < difficulty) {
        return Err(eyre!("sanity: CPU hash >= MAX difficulty"));
    }

    Ok(())
}

/// Read the 32-byte challenge as 4 little-endian u64 words.
fn split_challenge_le(challenge: &B256) -> [u64; 4] {
    let b = challenge.as_slice();
    [
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
        u64::from_le_bytes(b[16..24].try_into().unwrap()),
        u64::from_le_bytes(b[24..32].try_into().unwrap()),
    ]
}

/// Split a uint256 into 4 big-endian u64 words (index 0 = most significant).
fn split_difficulty_be(d: U256) -> [u64; 4] {
    let b = d.to_be_bytes::<32>();
    [
        u64::from_be_bytes(b[0..8].try_into().unwrap()),
        u64::from_be_bytes(b[8..16].try_into().unwrap()),
        u64::from_be_bytes(b[16..24].try_into().unwrap()),
        u64::from_be_bytes(b[24..32].try_into().unwrap()),
    ]
}

/// CPU reference for keccak256(abi.encode(bytes32 challenge, uint256 nonce)).
fn cpu_hash(challenge: &B256, nonce: U256) -> B256 {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge.as_slice());
    buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
    keccak256(buf)
}
