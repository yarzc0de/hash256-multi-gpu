# HASH Token GPU Miner (Rust) — Multi-GPU Multi-Wallet Edition

Fork dari [mrfunntastiic/hash256-cli-with-gpu](https://github.com/mrfunntastiic/hash256-cli-with-gpu) dengan **multi-GPU** dan **multi-wallet** support.

## Fitur

- ✅ **Multi-GPU**: Otomatis detect & pakai SEMUA GPU yang tersedia
- ✅ **Multi-Wallet**: `PRIVATE_KEY_1`, `PRIVATE_KEY_2`, ... — auto-assign GPU per wallet
- ✅ **Smart GPU Assignment**: 1 wallet = all GPUs, 2 wallets = 1 GPU each, dst
- ✅ **tmux Manager Script**: `./run.sh start|stop|restart|status|logs`
- ✅ **Backward compatible**: Single `PRIVATE_KEY` tetap jalan normal

## Quick Start

```bash
# Install Rust + OpenCL
apt update && apt install -y build-essential pkg-config ocl-icd-opencl-dev nvidia-opencl-dev clinfo git curl tmux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone & build
git clone https://github.com/yarzc0de/hash256-multi-gpu.git
cd hash256-multi-gpu
cargo build --release

# Config
cp .env.example .env
nano .env  # isi PRIVATE_KEY + config

# Run
./run.sh start
```

## Configuration (`.env`)

```env
# RPC endpoint
RPC_URL=https://ethereum-rpc.publicnode.com

# --- Wallets ---
# Multi-wallet (auto-assigns GPU per wallet)
PRIVATE_KEY_1=0xYOUR_FIRST_WALLET_KEY
PRIVATE_KEY_2=0xYOUR_SECOND_WALLET_KEY
# PRIVATE_KEY_3=0x...

# Single wallet (uses ALL GPUs)
# PRIVATE_KEY=0xYOUR_SINGLE_WALLET_KEY

# --- GPU ---
GPU=1
# GPU_BATCH=4194304        # nonces per dispatch per GPU (default: 4M)

# --- CPU fallback ---
MINER_THREADS=8

# --- Gas ---
PRIORITY_GWEI=20
MAX_FEE_GWEI=40
# GAS_LIMIT_OVERRIDE=200000
```

## GPU Assignment Logic

| Wallets | GPUs | Assignment |
|---------|------|------------|
| 1 wallet | 2 GPUs | Wallet gets ALL GPUs (max hashrate) |
| 2 wallets | 2 GPUs | Wallet 1 → GPU 0, Wallet 2 → GPU 1 |
| 2 wallets | 4 GPUs | Wallet 1 → GPU 0, Wallet 2 → GPU 1 |
| 3 wallets | 2 GPUs | Round-robin (wallet 3 shares GPU 0) |

## Manager Script (`run.sh`)

```bash
./run.sh start    # Start miner in background (tmux)
./run.sh stop     # Stop miner (Ctrl+C + kill)
./run.sh restart  # Stop + Start
./run.sh status   # Check if running + last output
./run.sh logs     # Attach to miner output (Ctrl+B, D to detach)
./run.sh build    # Rebuild binary
./run.sh update   # Git pull + rebuild
```

## Output Example (2 wallets, 2 GPUs)

```
🔐 HASH Token Miner (Rust) — Multi-GPU Multi-Wallet Edition
=============================================================

⛽ RPC: https://ethereum-rpc.publicnode.com
👛 Wallets: 2
🧵 CPU threads (fallback): 8
💸 Gas: priority=20 gwei, maxFee=40 gwei

📋 Mining State:
   Era: 0
   Reward: 100000000000000000000 (raw, 1e18)
   Difficulty: 26328072917139296674479506920917608079723773850137277813577744383
✅ Genesis complete — mining is open

🎮 GPUs detected: 2

🚀 Starting 2 mining worker(s)...
   Wallet 1 → GPU 0
   Wallet 2 → GPU 1

[W1] 📍 Address: 0xf1E3...8Fa5
[W1] 🎮 GPU: NVIDIA RTX PRO 6000 (1 device(s))
[W1] ✅ GPU self-test passed
[W2] 📍 Address: 0xa3f2...91c7
[W2] 🎮 GPU: NVIDIA RTX PRO 6000 (1 device(s))
[W2] ✅ GPU self-test passed

[W1] ⛏️  Epoch 250699 | GPU | 0x501bcaf7...
[W2] ⛏️  Epoch 250699 | GPU | 0xa3f2c891...
[W1] ⚡ 4700000000.00 H/s |   42s |  197568495616 attempts
[W2] ⚡ 4700000000.00 H/s |   42s |  197568495616 attempts
[W1] 🎉 NONCE FOUND: 8013143828958257601 (epoch 250699)
[W1] ✅ Confirmed block 25069901 | mints: 1
```

## Vast.ai Setup

1. Sewa instance dengan GPU (RTX 3090/4090/PRO 6000)
2. Pilih template PyTorch atau Ubuntu + CUDA
3. Buka Jupyter Terminal
4. Clone, build, config, run — semua GPU langsung kepake

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVATE_KEY` | (required) | Single wallet mode |
| `PRIVATE_KEY_1..N` | - | Multi-wallet mode (overrides PRIVATE_KEY) |
| `RPC_URL` | `https://eth.llamarpc.com` | Ethereum RPC endpoint |
| `GPU` | `0` | Set `1` to enable GPU mining |
| `GPU_BATCH` | `4194304` | Nonces per dispatch per GPU |
| `MINER_THREADS` | `num_cpus - 1` | CPU threads (fallback) |
| `PRIORITY_GWEI` | `5` | EIP-1559 priority fee |
| `MAX_FEE_GWEI` | `100` | EIP-1559 max fee ceiling |
| `GAS_LIMIT_OVERRIDE` | (auto) | Fixed gas limit |

## Why Multi-Wallet?

Challenge di contract HASH itu **per-address** — setiap wallet punya puzzle sendiri. Jadi 2 wallet × 4.7 GH/s menghasilkan **lebih banyak total mints** daripada 1 wallet × 9.4 GH/s, karena puzzle-nya independent.

## License

MIT
