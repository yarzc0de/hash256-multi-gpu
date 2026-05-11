use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use alloy::network::EthereumWallet;
use alloy::primitives::{address, keccak256, Address, B256, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use eyre::{eyre, Result};
use rand::Rng;

#[cfg(feature = "gpu")]
mod gpu;

const HASH_CONTRACT_ADDRESS: Address = address!("AC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc");
const DEFAULT_RPC_URL: &str = "https://eth.llamarpc.com";
const EPOCH_BLOCKS: u64 = 100;
const EPOCH_POLL_INTERVAL: Duration = Duration::from_secs(15);
const STATS_INTERVAL: Duration = Duration::from_secs(2);
const ERA_MINTS: u64 = 100_000;

sol! {
    #[sol(rpc)]
    contract HashToken {
        function currentDifficulty() external view returns (uint256);
        function totalMints() external view returns (uint256);
        function totalMiningMinted() external view returns (uint256);
        function genesisComplete() external view returns (bool);
        function getChallenge(address miner) external view returns (bytes32);
        function epochBlocksLeft() external view returns (uint256);
        function currentReward() external view returns (uint256);
        function miningState() external view returns (
            uint256 era,
            uint256 reward,
            uint256 difficulty,
            uint256 minted,
            uint256 remaining,
            uint256 epoch,
            uint256 epochBlocksLeft
        );
        function mine(uint256 nonce) external;
        function totalSupply() external view returns (uint256);
    }
}

struct Solution {
    nonce: U256,
    epoch: u64,
}

#[inline]
fn check_proof(challenge: &B256, nonce: U256, difficulty: U256) -> bool {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge.as_slice());
    buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
    let hash = keccak256(buf);
    U256::from_be_bytes::<32>(hash.0) < difficulty
}

fn run_workers(
    challenge: B256,
    difficulty: U256,
    epoch: u64,
    start_nonce: U256,
    stop_flag: Arc<AtomicBool>,
    attempts_counter: Arc<AtomicU64>,
    num_threads: usize,
) -> Option<Solution> {
    let solution_slot: Mutex<Option<Solution>> = Mutex::new(None);
    let stride = U256::from(num_threads);

    std::thread::scope(|s| {
        for tid in 0..num_threads {
            let stop_flag = &stop_flag;
            let attempts_counter = &attempts_counter;
            let solution_slot = &solution_slot;
            s.spawn(move || {
                let mut nonce = start_nonce + U256::from(tid);
                let mut local_attempts: u64 = 0;
                loop {
                    if check_proof(&challenge, nonce, difficulty) {
                        let mut slot = solution_slot.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(Solution { nonce, epoch });
                        }
                        stop_flag.store(true, Ordering::Relaxed);
                        attempts_counter.fetch_add(local_attempts, Ordering::Relaxed);
                        return;
                    }
                    nonce += stride;
                    local_attempts += 1;

                    if local_attempts & 0x3FFF == 0 {
                        attempts_counter.fetch_add(local_attempts, Ordering::Relaxed);
                        local_attempts = 0;
                        if stop_flag.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                }
            });
        }
    });

    solution_slot.into_inner().ok().flatten()
}

fn reward_for_total_mints(total_mints: U256) -> u128 {
    let era = total_mints / U256::from(ERA_MINTS);
    if era < U256::from(64u64) {
        100u128 >> era.to::<u128>().min(63)
    } else {
        0
    }
}

fn hex_short(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes.iter().take(8) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Collect private keys from env: PRIVATE_KEY_1, PRIVATE_KEY_2, ... or fallback to PRIVATE_KEY
fn collect_private_keys() -> Result<Vec<String>> {
    let mut keys = Vec::new();

    // Try PRIVATE_KEY_1, PRIVATE_KEY_2, ...
    for i in 1..=32 {
        let var_name = format!("PRIVATE_KEY_{}", i);
        match std::env::var(&var_name) {
            Ok(v) if !v.trim().is_empty() => keys.push(v),
            _ => break,
        }
    }

    // Fallback to single PRIVATE_KEY
    if keys.is_empty() {
        match std::env::var("PRIVATE_KEY") {
            Ok(v) if !v.trim().is_empty() => keys.push(v),
            _ => {
                println!("⚠️  No PRIVATE_KEY env var found.");
                let k = rpassword::prompt_password("Private Key: ")?;
                keys.push(k);
            }
        }
    }

    Ok(keys)
}

/// Run mining loop for a single wallet with assigned GPU index
async fn mine_wallet(
    wallet_idx: usize,
    raw_key: String,
    rpc_url_str: String,
    gpu_index: Option<usize>,
    num_threads: usize,
    shutdown: Arc<AtomicBool>,
    gpu_batch: Option<usize>,
    priority_gwei: f64,
    max_fee_gwei: f64,
    gas_limit_override: Option<u64>,
) -> Result<(u64, u64)> {
    let key_trimmed = raw_key.trim().trim_start_matches("0x");
    if key_trimmed.len() != 64 {
        return Err(eyre!("Wallet {}: Invalid private key length", wallet_idx));
    }
    let signer: PrivateKeySigner = key_trimmed.parse()?;
    let miner_address = signer.address();
    let wallet = EthereumWallet::from(signer);

    let provider = ProviderBuilder::new()
        .with_recommended_fillers()
        .wallet(wallet)
        .on_http(rpc_url_str.parse()?);

    let contract = HashToken::new(HASH_CONTRACT_ADDRESS, provider.clone());

    let tag = format!("[W{}]", wallet_idx);
    println!("{} 📍 Address: {}", tag, miner_address);

    // --- GPU setup for this wallet ---
    #[cfg(feature = "gpu")]
    let gpu_miner: Option<Arc<gpu::GpuMiner>> = if gpu_index.is_some() || std::env::var("GPU").ok().as_deref() == Some("1") {
        match gpu::GpuMiner::new(gpu_batch, gpu_index) {
            Ok(g) => {
                println!("{} 🎮 GPU: {} ({} device(s))", tag, g.device_name(), g.device_count());
                match g.self_test() {
                    Ok(()) => println!("{} ✅ GPU self-test passed", tag),
                    Err(e) => {
                        eprintln!("{} ⚠️  GPU self-test failed: {e}, falling back to CPU", tag);
                        None?;
                        unreachable!()
                    }
                }
                Some(Arc::new(g))
            }
            Err(e) => {
                eprintln!("{} ⚠️  GPU init failed: {e}, using CPU", tag);
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_miner: Option<()> = None;

    let mut session_attempts: u64 = 0;
    let mut success_count: u64 = 0;

    while !shutdown.load(Ordering::Relaxed) {
        let block_num = match provider.get_block_number().await {
            Ok(n) => n,
            Err(e) => {
                eprintln!("{} ❌ RPC error (block): {e}", tag);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        let epoch: u64 = block_num / EPOCH_BLOCKS;

        let challenge = match contract.getChallenge(miner_address).call().await {
            Ok(v) => v._0,
            Err(e) => {
                eprintln!("{} ❌ RPC error (challenge): {e}", tag);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let difficulty = match contract.currentDifficulty().call().await {
            Ok(v) => v._0,
            Err(e) => {
                eprintln!("{} ❌ RPC error (difficulty): {e}", tag);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let backend = if gpu_miner.is_some() { "GPU" } else { "CPU" };
        println!("{} ⛏️  Epoch {} | {} | 0x{}...", tag, epoch, backend, hex_short(challenge.as_slice()));

        let start_nonce_u64: u64 = rand::thread_rng().gen();
        let start_nonce = U256::from(start_nonce_u64);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let attempts_counter = Arc::new(AtomicU64::new(0));

        // Watchdog: epoch poll
        let watchdog = {
            let stop_flag = Arc::clone(&stop_flag);
            let attempts_counter = Arc::clone(&attempts_counter);
            let shutdown = Arc::clone(&shutdown);
            let provider = provider.clone();
            let target_epoch = epoch;
            let round_start = Instant::now();
            let tag = tag.clone();
            tokio::spawn(async move {
                let mut last_print = Instant::now();
                let mut last_attempts: u64 = 0;
                let mut last_poll = Instant::now();
                loop {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if stop_flag.load(Ordering::Relaxed) { break; }
                    if shutdown.load(Ordering::Relaxed) {
                        stop_flag.store(true, Ordering::Relaxed);
                        break;
                    }

                    if last_print.elapsed() >= STATS_INTERVAL {
                        let total = attempts_counter.load(Ordering::Relaxed);
                        let delta = total.saturating_sub(last_attempts);
                        let secs = last_print.elapsed().as_secs_f64().max(0.001);
                        let rate = delta as f64 / secs;
                        let elapsed = round_start.elapsed().as_secs_f64();
                        eprint!(
                            "\r{} ⚡ {:>10.2} H/s | {:>5.0}s | {:>12} attempts",
                            tag, rate, elapsed, total
                        );
                        last_attempts = total;
                        last_print = Instant::now();
                    }

                    if last_poll.elapsed() >= EPOCH_POLL_INTERVAL {
                        last_poll = Instant::now();
                        match provider.get_block_number().await {
                            Ok(bn) => {
                                let cur_epoch = bn / EPOCH_BLOCKS;
                                if cur_epoch != target_epoch {
                                    eprintln!(
                                        "\n{} 🔄 Epoch {} -> {} (block {})",
                                        tag, target_epoch, cur_epoch, bn
                                    );
                                    stop_flag.store(true, Ordering::Relaxed);
                                    break;
                                }
                            }
                            Err(e) => eprintln!("\n{} ⚠️  poll failed: {e}", tag),
                        }
                    }
                }
            })
        };

        // Mining
        let mining_result: Option<Solution> = {
            let stop_flag = Arc::clone(&stop_flag);
            let attempts_counter = Arc::clone(&attempts_counter);

            #[cfg(feature = "gpu")]
            {
                if let Some(g) = gpu_miner.as_ref().cloned() {
                    let res = tokio::task::spawn_blocking(move || {
                        g.mine(challenge, difficulty, start_nonce_u64, stop_flag, attempts_counter)
                    })
                    .await?;
                    match res {
                        Ok(Some(nonce_u64)) => Some(Solution {
                            nonce: U256::from(nonce_u64),
                            epoch,
                        }),
                        Ok(None) => None,
                        Err(e) => {
                            eprintln!("{} ❌ GPU error: {e}", tag);
                            None
                        }
                    }
                } else {
                    tokio::task::spawn_blocking(move || {
                        run_workers(
                            challenge, difficulty, epoch, start_nonce,
                            stop_flag, attempts_counter, num_threads,
                        )
                    })
                    .await?
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                let _ = &gpu_miner;
                tokio::task::spawn_blocking(move || {
                    run_workers(
                        challenge, difficulty, epoch, start_nonce,
                        stop_flag, attempts_counter, num_threads,
                    )
                })
                .await?
            }
        };

        stop_flag.store(true, Ordering::Relaxed);
        let _ = watchdog.await;
        let round_attempts = attempts_counter.load(Ordering::Relaxed);
        session_attempts += round_attempts;
        eprintln!();

        let Some(sol) = mining_result else {
            continue;
        };

        println!("{} 🎉 NONCE FOUND: {} (epoch {})", tag, sol.nonce, sol.epoch);

        let priority_wei = (priority_gwei * 1e9) as u128;
        let max_fee_wei = (max_fee_gwei * 1e9) as u128;

        let mut tx = contract
            .mine(sol.nonce)
            .max_priority_fee_per_gas(priority_wei)
            .max_fee_per_gas(max_fee_wei);
        if let Some(g) = gas_limit_override {
            tx = tx.gas(g);
        }

        match tx.send().await {
            Ok(pending) => {
                let tx_hash = *pending.tx_hash();
                println!("{} 📋 TX: {tx_hash}", tag);
                match pending.with_required_confirmations(1).get_receipt().await {
                    Ok(receipt) => {
                        if receipt.status() {
                            success_count += 1;
                            println!(
                                "{} ✅ Confirmed block {} | mints: {}",
                                tag,
                                receipt.block_number.unwrap_or_default(),
                                success_count
                            );
                        } else {
                            println!("{} ❌ Reverted", tag);
                        }
                    }
                    Err(e) => eprintln!("{} ❌ Receipt error: {e}", tag),
                }
            }
            Err(e) => eprintln!("{} ❌ Submit failed: {e}", tag),
        }
    }

    Ok((session_attempts, success_count))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    println!("🔐 HASH Token Miner (Rust) — Multi-GPU Multi-Wallet Edition");
    println!("=============================================================\n");

    let keys = collect_private_keys()?;
    let num_wallets = keys.len();

    let rpc_url_str = std::env::var("RPC_URL").unwrap_or_else(|_| DEFAULT_RPC_URL.to_string());
    let num_threads = std::env::var("MINER_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1));
    let gpu_batch = std::env::var("GPU_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let priority_gwei: f64 = std::env::var("PRIORITY_GWEI")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let max_fee_gwei: f64 = std::env::var("MAX_FEE_GWEI")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100.0);
    let gas_limit_override: Option<u64> = std::env::var("GAS_LIMIT_OVERRIDE")
        .ok()
        .and_then(|s| s.parse().ok());

    println!("⛽ RPC: {}", rpc_url_str);
    println!("👛 Wallets: {}", num_wallets);
    println!("🧵 CPU threads (fallback): {}", num_threads);
    println!("💸 Gas: priority={} gwei, maxFee={} gwei", priority_gwei, max_fee_gwei);

    // --- Show mining state ---
    {
        let provider = ProviderBuilder::new()
            .on_http(rpc_url_str.parse()?);
        let contract = HashToken::new(HASH_CONTRACT_ADDRESS, provider.clone());
        match contract.miningState().call().await {
            Ok(s) => {
                println!("\n📋 Mining State:");
                println!("   Era: {}", s.era);
                println!("   Reward: {} (raw, 1e18)", s.reward);
                println!("   Difficulty: {}", s.difficulty);
                println!("   Minted: {} / {}", s.minted, s.minted + s.remaining);
                println!("   Epoch: {} | Blocks left: {}", s.epoch, s.epochBlocksLeft);
            }
            Err(e) => eprintln!("⚠️  miningState() failed: {e}"),
        }
        match contract.genesisComplete().call().await {
            Ok(g) if !g._0 => {
                return Err(eyre!("Genesis not complete — mining closed."));
            }
            Ok(_) => println!("✅ Genesis complete — mining is open"),
            Err(e) => eprintln!("⚠️  genesisComplete check failed: {e}"),
        }
    }

    // --- Detect GPUs ---
    #[cfg(feature = "gpu")]
    let total_gpus = {
        use ocl::{Device, Platform};
        let platform = Platform::default();
        Device::list_all(platform).map(|d| d.len()).unwrap_or(0)
    };
    #[cfg(not(feature = "gpu"))]
    let total_gpus = 0usize;

    println!("\n🎮 GPUs detected: {}", total_gpus);

    // --- Assign GPUs to wallets ---
    // Strategy: distribute GPUs evenly across wallets
    // e.g. 2 wallets, 4 GPUs → wallet 0 gets GPU 0,1 — wallet 1 gets GPU 2,3
    // e.g. 2 wallets, 2 GPUs → wallet 0 gets GPU 0 — wallet 1 gets GPU 1
    // e.g. 1 wallet, 2 GPUs → wallet 0 gets all GPUs (gpu_index=None)
    let gpu_assignments: Vec<Option<usize>> = if total_gpus == 0 || num_wallets == 0 {
        vec![None; num_wallets]
    } else if num_wallets == 1 {
        // Single wallet gets ALL GPUs
        vec![None]
    } else if num_wallets >= total_gpus {
        // More wallets than GPUs: each wallet gets 1 GPU (round-robin)
        (0..num_wallets).map(|i| Some(i % total_gpus)).collect()
    } else {
        // More GPUs than wallets: each wallet gets 1 GPU (for now)
        // TODO: could split GPU ranges per wallet
        (0..num_wallets).map(|i| Some(i)).collect()
    };

    println!("\n🚀 Starting {} mining worker(s)...", num_wallets);
    for (i, gpu) in gpu_assignments.iter().enumerate() {
        match gpu {
            Some(idx) => println!("   Wallet {} → GPU {}", i + 1, idx),
            None => println!("   Wallet {} → All GPUs", i + 1),
        }
    }
    println!();

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                println!("\n🛑 Ctrl-C received, stopping all workers...");
                shutdown.store(true, Ordering::Relaxed);
            }
        });
    }

    let session_start = Instant::now();

    // Spawn a task per wallet
    let mut handles = Vec::new();
    for (i, key) in keys.into_iter().enumerate() {
        let rpc = rpc_url_str.clone();
        let shutdown = Arc::clone(&shutdown);
        let gpu_idx = gpu_assignments[i];

        let handle = tokio::spawn(mine_wallet(
            i + 1,
            key,
            rpc,
            gpu_idx,
            num_threads,
            shutdown,
            gpu_batch,
            priority_gwei,
            max_fee_gwei,
            gas_limit_override,
        ));
        handles.push(handle);
    }

    // Wait for all workers
    let mut total_attempts: u64 = 0;
    let mut total_mints: u64 = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok((attempts, mints))) => {
                total_attempts += attempts;
                total_mints += mints;
                println!("[W{}] Done — {} attempts, {} mints", i + 1, attempts, mints);
            }
            Ok(Err(e)) => eprintln!("[W{}] Error: {e}", i + 1),
            Err(e) => eprintln!("[W{}] Task panic: {e}", i + 1),
        }
    }

    let elapsed = session_start.elapsed().as_secs_f64().max(0.001);
    let rate = total_attempts as f64 / elapsed;
    println!("\n📊 Final Statistics:");
    println!("   Total Attempts: {total_attempts}");
    println!("   Total Mints: {total_mints}");
    println!("   Combined Hash Rate: {rate:.2} H/s");
    println!("   Duration: {elapsed:.2}s");
    Ok(())
}
