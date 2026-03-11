#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print('CUDA is not available. Skipping NB201 CUDA runtime benchmark.')
    sys.exit(1)
PY

REPEATS="${REPEATS:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"

DATA_PATH="${DATA_PATH:-/data/gpfs/projects/punim1875/arch2vec-readonly/data/cifar10_valid_converged.json}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LATENT_DIM="${LATENT_DIM:-16}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"

OUT_BASE="${OUT_BASE:-/data/gpfs/projects/punim1875/arch2vec-readonly/results/runtime_report_nb201_arch2vec/cuda}"
RAW_DIR="$OUT_BASE/raw"
mkdir -p "$RAW_DIR"

count=0
for seed in $SEEDS; do
  if [ "$count" -ge "$REPEATS" ]; then
    break
  fi
  profile_csv="$RAW_DIR/profile_seed_${seed}.csv"
  echo "[$((count+1))/$REPEATS] seed=$seed -> $profile_csv"
  python models/pretraining_nasbench201.py \
    --device "$DEVICE" \
    --seed "$seed" \
    --data "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --bs "$BATCH_SIZE" \
    --latent_dim "$LATENT_DIM" \
    --lr "$LR" \
    --profile \
    --profile_sync \
    --profile_out "$profile_csv"
  count=$((count+1))
done

echo "NB201 CUDA runtime runs complete. Raw profiles in $RAW_DIR"
