#!/bin/bash
# Like combos.sbatch, but runs several model queues ("lanes") at once, one per GPU of a
# multi-GPU allocation. Lanes are separated by "|":
#
#   LANES="qwen3-32b qwen2.5-7b|qwen2.5-32b qwen3-8b|qwen2.5-14b qwen3-1.7b" \
#     sbatch -p gpu-interactive --gres=gpu:a100:3 bench/lanes.sbatch
#
# Each lane pins its server and client to one GPU with CUDA_VISIBLE_DEVICES; the lanes
# share the node's CPUs, so ask for enough of them (default 48).
#SBATCH --job-name=gpufc-lanes
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=01:59:00
#SBATCH --output=/scratch/%u/gpufc/logs/%x-%j.log
set -uo pipefail
ROOT=${GPUFC_ROOT:-/scratch/$USER/gpufc}
LANES=${LANES:?set LANES}
MIN_LEFT=${MIN_LEFT:-22}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
module load apptainer 2>/dev/null || true
hostname; nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | cut -d. -f1)
export GPUFC_COMPAT=""
(( DRIVER_MAJOR < 580 )) && GPUFC_COMPAT=/usr/local/cuda/compat
if (( DRIVER_MAJOR != 570 && DRIVER_MAJOR < 580 )); then
  echo "driver $DRIVER_MAJOR on $(hostname) cannot run the CUDA 13 image (needs 570 or 580+)"; exit 3
fi
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c '
  export LD_LIBRARY_PATH=${GPUFC_COMPAT:+$GPUFC_COMPAT:}${LD_LIBRARY_PATH:-}
  python3 "$0/bench/jobtools.py" microbench --out "$1"' "$ROOT/repo" "$ROOT/gpu-${SLURM_JOB_ID}.json" \
  || echo "microbench failed"

left_min() {
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

IFS='|' read -r -a LANE_LIST <<<"$LANES"
for i in "${!LANE_LIST[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=$i
    for m in ${LANE_LIST[$i]}; do
      left=$(left_min)
      if (( left < MIN_LEFT )); then echo "[lane $i] skip $m: only ${left} min left"; continue; fi
      echo "[lane $i] #### $m ($(date +%T), ${left} min left)"
      timeout $(( (left - 2) * 60 )) bash "$ROOT/repo/bench/run_combo.sh" "$m" 2>&1 \
        | sed -u "s/^/[lane $i] /" || echo "[lane $i] #### $m failed"
    done
  ) &
  sleep 20  # stagger server start-up (weights load from shared storage)
done
wait
echo "all done $(date +%T)"
