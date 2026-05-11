//! GPU mining backend via OpenCL with multi-device support.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use alloy::primitives::{keccak256, B256, U256};
use eyre::{eyre, Result};
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue, SpatialDims};

const KERNEL_SRC: &str = include_str!("keccak_kernel.cl");

const DEFAULT_BATCH: usize = 1 << 24;

const DEFAULT_LWS: usize = 128;

const ARG_NONCE_BASE: u32 = 8;

struct GpuDevice {
    #[allow(dead_code)]
    context: Context,
    queue: Queue,
    program: Program,
    device_name: String,
    batch_size: usize,
    lws: usize,
}

pub struct GpuMiner {
    devices: Vec<GpuDevice>,
    batch_size: usize,
}

impl GpuMiner {
    pub fn new(batch_size: Option<usize>, gpu_indices: Option<Vec<usize>>) -> Result<Self> {
        // GPU_LWS overrides work-group size for tuning experiments.
        // Must divide batch_size; we round up to satisfy that.
        let lws = std::env::var("GPU_LWS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0 && v <= 1024)
            .unwrap_or(DEFAULT_LWS);

        let raw_batch = batch_size.unwrap_or(DEFAULT_BATCH);
        let batch_size = raw_batch.div_ceil(lws) * lws;

        // USE_MANUAL_ROL64=1 switches the kernel from rotate() builtin back to
        // manual shift+or ROL64. Escape hatch for drivers that mis-lower the
        // 64-bit rotate builtin (some NVIDIA OpenCL versions).
        let use_manual_rol = std::env::var("USE_MANUAL_ROL64").ok().as_deref() == Some("1");
        let mut cmplr_opts =
            String::from("-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math");
        if use_manual_rol {
            cmplr_opts.push_str(" -DUSE_MANUAL_ROL64");
        }

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
                // `-cl-fast-relaxed-math` relaxes FP rounding semantics; the kernel
                // is integer-only, so this is a zero-semantics change but lets
                // NVIDIA's OpenCL compiler emit tighter PTX on the int paths too.
                .cmplr_opt(cmplr_opts.as_str())
                .build(&context)?;

            devices.push(GpuDevice {
                context,
                queue,
                program,
                device_name,
                batch_size,
                lws,
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

    pub fn self_test(&self) -> Result<()> {
        for (i, dev) in self.devices.iter().enumerate() {
            self_test_device(dev)
                .map_err(|e| eyre!("GPU {} ({}) self-test failed: {e}", i, dev.device_name))?;
        }
        Ok(())
    }

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

        if num_gpus == 1 {
            return mine_on_device(
                &self.devices[0],
                challenge,
                difficulty,
                start_nonce,
                1,
                stop_flag,
                attempts_counter,
            );
        }

        let result: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);
        let batch = self.batch_size as u64;
        let stride = num_gpus as u64;

        std::thread::scope(|s| {
            for (gpu_idx, dev) in self.devices.iter().enumerate() {
                let stop_flag = &stop_flag;
                let attempts_counter = &attempts_counter;
                let result = &result;
                let gpu_start = start_nonce.wrapping_add((gpu_idx as u64) * batch);

                s.spawn(move || {
                    match mine_on_device(
                        dev,
                        challenge,
                        difficulty,
                        gpu_start,
                        stride,
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

fn mine_on_device(
    dev: &GpuDevice,
    challenge: B256,
    difficulty: U256,
    start_nonce: u64,
    stride_multiplier: u64,
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

    let kernel = Kernel::builder()
        .program(&dev.program)
        .name("mine_keccak")
        .queue(dev.queue.clone())
        .global_work_size(SpatialDims::One(dev.batch_size))
        .local_work_size(SpatialDims::One(dev.lws))
        .arg(cw[0])
        .arg(cw[1])
        .arg(cw[2])
        .arg(cw[3])
        .arg(dw[0])
        .arg(dw[1])
        .arg(dw[2])
        .arg(dw[3])
        .arg(0u64)
        .arg(&found_nonce)
        .arg(&found_flag)
        .build()?;

    let advance = (dev.batch_size as u64).saturating_mul(stride_multiplier);
    let mut nonce_base: u64 = start_nonce;
    let batch_u64 = dev.batch_size as u64;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return Ok(None);
        }

        found_flag.write(&[0i32][..]).enq()?;

        kernel.set_arg(ARG_NONCE_BASE, nonce_base)?;

        unsafe {
            kernel.enq()?;
        }
        dev.queue.finish()?;

        attempts_counter.fetch_add(batch_u64, Ordering::Relaxed);

        let mut flag = [0i32];
        found_flag.read(&mut flag[..]).enq()?;
        if flag[0] != 0 {
            let mut got = [0u64];
            found_nonce.read(&mut got[..]).enq()?;
            let nonce = got[0];

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

        nonce_base = nonce_base.wrapping_add(advance);
    }
}

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

    // Leaving LWS unset here: a 1-item GWS with LWS=128 is illegal under
    // clEnqueueNDRangeKernel.
    let kernel = Kernel::builder()
        .program(&dev.program)
        .name("mine_keccak")
        .queue(dev.queue.clone())
        .global_work_size(1usize)
        .arg(cw[0])
        .arg(cw[1])
        .arg(cw[2])
        .arg(cw[3])
        .arg(dw[0])
        .arg(dw[1])
        .arg(dw[2])
        .arg(dw[3])
        .arg(nonce_base)
        .arg(&found_nonce)
        .arg(&found_flag)
        .build()?;

    unsafe {
        kernel.enq()?;
    }
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

fn split_challenge_le(challenge: &B256) -> [u64; 4] {
    let b = challenge.as_slice();
    [
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
        u64::from_le_bytes(b[16..24].try_into().unwrap()),
        u64::from_le_bytes(b[24..32].try_into().unwrap()),
    ]
}

fn split_difficulty_be(d: U256) -> [u64; 4] {
    let b = d.to_be_bytes::<32>();
    [
        u64::from_be_bytes(b[0..8].try_into().unwrap()),
        u64::from_be_bytes(b[8..16].try_into().unwrap()),
        u64::from_be_bytes(b[16..24].try_into().unwrap()),
        u64::from_be_bytes(b[24..32].try_into().unwrap()),
    ]
}

fn cpu_hash(challenge: &B256, nonce: U256) -> B256 {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge.as_slice());
    buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
    keccak256(buf)
}
