#!/bin/bash
# Run bench/run_combo.sh for several models, one after another, on one GPU.
#
#   MODELS="qwen3-0.6b qwen3-1.7b" sbatch -p gpu-short --gres=gpu:h200:1 bench/combos.sbatch
#
# A model is skipped when less than MIN_LEFT minutes remain in the allocation.
#SBATCH --job-name=gpufc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:59:00
#SBATCH --output=/scratch/%u/gpufc/logs/%x-%j.log
set -uo pipefail
ROOT=${GPUFC_ROOT:-/scratch/$USER/gpufc}
MODELS=${MODELS:?set MODELS}
MIN_LEFT=${MIN_LEFT:-22}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
module load apptainer 2>/dev/null || true
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | cut -d. -f1)
export GPUFC_COMPAT=""
(( DRIVER_MAJOR < 580 )) && GPUFC_COMPAT=/usr/local/cuda/compat
if (( DRIVER_MAJOR != 570 && DRIVER_MAJOR < 580 )); then
  echo "driver $DRIVER_MAJOR on $(hostname) cannot run the CUDA 13 image (needs 570 or 580+)"; exit 3
fi
apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c '
  export LD_LIBRARY_PATH=${GPUFC_COMPAT:+$GPUFC_COMPAT:}${LD_LIBRARY_PATH:-}
  python3 "$0/bench/jobtools.py" microbench --out "$1"' "$ROOT/repo" "$ROOT/gpu-${SLURM_JOB_ID}.json" \
  || echo "microbench failed"

left_min() {  # minutes left in this allocation
  local l; l=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
  python3 - "$l" <<'PY'
import sys
s = sys.argv[1].strip(); d = 0
if "-" in s: d, s = s.split("-"); d = int(d)
p = [int(x) for x in s.split(":")]
while len(p) < 3: p.insert(0, 0)
print(d * 1440 + p[0] * 60 + p[1])
PY
}

for m in $MODELS; do
  left=$(left_min)
  if (( left < MIN_LEFT )); then echo "skip $m: only ${left} min left"; continue; fi
  echo "#### $m ($(date +%T), ${left} min left)"
  timeout $(( (left - 2) * 60 )) bash "$ROOT/repo/bench/run_combo.sh" "$m" || echo "#### $m failed (rc=$?)"
done
echo "all done $(date +%T)"
