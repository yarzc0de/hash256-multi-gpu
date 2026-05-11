# HASH Token GPU Miner (Rust) — Multi-GPU Edition

Fork dari [mrfunntastiic/hash256-cli-with-gpu](https://github.com/mrfunntastiic/hash256-cli-with-gpu) dengan **multi-GPU support** — semua GPU di sistem langsung kepake parallel.

## Fitur Tambahan

- ✅ **Multi-GPU**: Otomatis detect & pakai SEMUA GPU yang tersedia
- ✅ **Per-GPU threading**: Setiap GPU jalan di thread terpisah, nonce range berbeda
- ✅ **GPU_INDEX**: Opsional pilih GPU tertentu saja
- ✅ **Backward compatible**: Single GPU tetap jalan normal tanpa overhead

## Quick Start

```bash
# Install Rust + OpenCL
apt update && apt install -y build-essential pkg-config ocl-icd-opencl-dev nvidia-opencl-dev clinfo git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone & build
git clone https://github.com/yarzc0de/hash256-multi-gpu.git
cd hash256-multi-gpu
cargo build --release

# Config
cp .env.example .env
nano .env  # isi PRIVATE_KEY

# Run (semua GPU otomatis kepake)
./target/release/hash-miner-rs
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVATE_KEY` | (required) | Wallet private key (0x...) |
| `RPC_URL` | `https://eth.llamarpc.com` | Ethereum RPC endpoint |
| `GPU` | `1` | Set `1` to enable GPU mining |
| `GPU_INDEX` | (all) | Use specific GPU only (0-based index) |
| `GPU_BATCH` | `4194304` | Nonces per dispatch per GPU |
| `MINER_THREADS` | (all cores) | CPU threads (fallback if no GPU) |
| `PRIORITY_GWEI` | `5` | EIP-1559 priority fee |
| `MAX_FEE_GWEI` | `100` | EIP-1559 max fee ceiling |
| `GAS_LIMIT_OVERRIDE` | (auto) | Fixed gas limit |

## Multi-GPU Usage

```bash
# Pakai SEMUA GPU (default)
GPU=1 ./target/release/hash-miner-rs

# Pakai GPU tertentu saja (index 0)
GPU=1 GPU_INDEX=0 ./target/release/hash-miner-rs

# Custom batch size per GPU
GPU=1 GPU_BATCH=8388608 ./target/release/hash-miner-rs
```

## Vast.ai Setup

1. Sewa instance dengan multi-GPU (e.g. 4x RTX 3090)
2. Pilih template PyTorch atau Ubuntu + CUDA
3. SSH masuk, install deps, clone, build, run
4. Semua GPU langsung kepake — no manual config needed

## Output Example (4x GPU)

```
🔐 HASH Token Miner (Rust) — Multi-GPU Edition
================================================

🎮 GPU Mining — 4 device(s) detected:
   GPU 0: NVIDIA GeForce RTX 3090
   GPU 1: NVIDIA GeForce RTX 3090
   GPU 2: NVIDIA GeForce RTX 3090
   GPU 3: NVIDIA GeForce RTX 3090
   Batch size: 4194304 nonces/dispatch/GPU
✅ All GPU self-tests passed

⛏️  Mining epoch 1234 on GPU (4 GPUs)...
⚡ 22000000000.00 H/s | round   12s | attempts    264000000000
```

## License

MIT
