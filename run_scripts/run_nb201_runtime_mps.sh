#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import torch, sys
if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
    print('MPS is not available. Skipping NB201 MPS runtime benchmark.')
    sys.exit(1)
PY

REPEATS="${REPEATS:-5}"
SEEDS="${SEEDS:-10 20 30 40 50}"

DATA_PATH="${DATA_PATH:-data/cifar10_valid_converged.json}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LATENT_DIM="${LATENT_DIM:-16}"
LR="${LR:-1e-3}"

OUT_BASE="${OUT_BASE:-results/runtime_report_nb201_arch2vec/mps}"
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
    --device mps \
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

echo "NB201 MPS runtime runs complete. Raw profiles in $RAW_DIR"
