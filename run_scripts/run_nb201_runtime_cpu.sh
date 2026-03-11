#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPEATS="${REPEATS:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"

DATA_PATH="${DATA_PATH:-data/cifar10_valid_converged.json}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LATENT_DIM="${LATENT_DIM:-16}"
LR="${LR:-1e-3}"

OUT_BASE="${OUT_BASE:-results/runtime_report_nb201_arch2vec/cpu}"
RAW_DIR="$OUT_BASE/raw"
mkdir -p "$RAW_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

count=0
for seed in $SEEDS; do
  if [ "$count" -ge "$REPEATS" ]; then
    break
  fi
  profile_csv="$RAW_DIR/profile_seed_${seed}.csv"
  echo "[$((count+1))/$REPEATS] seed=$seed -> $profile_csv"
  python models/pretraining_nasbench201.py \
    --device cpu \
    --cpu_threads "${OMP_NUM_THREADS}" \
    --seed "$seed" \
    --data "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --bs "$BATCH_SIZE" \
    --latent_dim "$LATENT_DIM" \
    --lr "$LR" \
    --profile \
    --profile_out "$profile_csv"
  count=$((count+1))
done

echo "NB201 CPU runtime runs complete. Raw profiles in $RAW_DIR"
