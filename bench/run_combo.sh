#!/bin/bash
# Profile one model on the GPU this Slurm job holds, then replay validation traces.
#
#   bench/run_combo.sh qwen3-8b
#
# Layout under $GPUFC_ROOT (default /scratch/$USER/gpufc):
#   repo/       this repository (rsynced)       batchsim/  batchsim sources
#   data/       Azure trace CSVs                runs/<gpu>/<model>/<job>/  outputs
#
# Output per run: meta.json, gpu.json (microbenchmark), server.log, steps.jsonl (probe),
# runs.jsonl (replay windows), <run>.csv (workload) and <run>.jsonl (client records).
# The server runs in its own session and only its process group is ever signalled:
# GPU nodes are shared with other jobs of the same user.
set -uo pipefail
MODEL_KEY=${1:?usage: run_combo.sh <model-key>}
ROOT=${GPUFC_ROOT:-/scratch/$USER/gpufc}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
REPO=$ROOT/repo
export HF_HOME=${HF_HOME:-/scratch/$USER/hf} HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}
bash "$REPO/bench/install_probe.sh" "$ROOT/probe" >/dev/null
export PYTHONPATH=$ROOT/probe:$REPO/src:$ROOT/batchsim/src
module load apptainer 2>/dev/null || true
# The image is built for CUDA 13. Drivers older than 580 need its forward-compatibility
# libraries (supported on long-term branches such as 570, not on 545); 580+ must not use them.
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | cut -d. -f1)
export GPUFC_COMPAT=""
(( DRIVER_MAJOR < 580 )) && GPUFC_COMPAT=/usr/local/cuda/compat

in_image() {
  apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c '
    export LD_LIBRARY_PATH=${GPUFC_COMPAT:+$GPUFC_COMPAT:}${LD_LIBRARY_PATH:-}
    exec "$@"' _ "$@"
}
jt() { in_image python3 "$REPO/bench/jobtools.py" "$@"; }

HF_ID=$(python3 "$REPO/bench/jobtools.py" hf-id "$MODEL_KEY") || exit 2
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1)
GPU_KEY=$(echo "$GPU_NAME" | tr 'A-Z ' 'a-z-' | sed 's/^nvidia-//;s/^tesla-//')
# vLLM 0.34 OpenAI-server defaults, passed explicitly so every run records them.
if (( GPU_MIB >= 70 * 1024 )) && [[ $GPU_NAME != *A100* ]]; then BT=8192; NS=1024; else BT=2048; NS=256; fi
DTYPE=bfloat16; [[ ${CC%%.*} -lt 8 ]] && DTYPE=float16
MAXLEN=8192
OUT=$ROOT/runs/$GPU_KEY/$MODEL_KEY/${SLURM_JOB_ID:-local}-$(date +%H%M%S)
mkdir -p "$OUT" "$ROOT/cache"
export BATCHSIM_STEP_LOG=$OUT/steps.jsonl
echo "== $MODEL_KEY ($HF_ID) on $GPU_NAME -> $OUT"

if [ -f "$ROOT/gpu-${SLURM_JOB_ID:-local}.json" ]; then cp "$ROOT/gpu-${SLURM_JOB_ID:-local}.json" "$OUT/gpu.json"; fi
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem,power.limit,compute_cap --format=csv,noheader -i 0 >"$OUT/nvidia-smi.txt"
cat >"$OUT/meta.json" <<EOF
{"model": "$MODEL_KEY", "hf_id": "$HF_ID", "gpu_name": "$GPU_NAME", "gpu_key": "$GPU_KEY",
 "dtype": "$DTYPE", "max_num_batched_tokens": $BT, "max_num_seqs": $NS, "max_model_len": $MAXLEN,
 "vllm_args": "--max-num-batched-tokens $BT --max-num-seqs $NS --max-model-len $MAXLEN --generation-config vllm",
 "job": "${SLURM_JOB_ID:-local}", "node": "$(hostname)", "cpu": "$(lscpu | sed -n 's/^Model name: *//p' | head -1)",
 "sif": "$SIF"}
EOF

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
URL=http://127.0.0.1:$PORT
SERVER=""
cat >"$OUT/serve.sh" <<EOF
#!/bin/bash
exec apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c '
  export LD_LIBRARY_PATH=\${GPUFC_COMPAT:+\$GPUFC_COMPAT:}\${LD_LIBRARY_PATH:-}
  exec vllm serve "$HF_ID" --host 127.0.0.1 --port $PORT --served-model-name m --dtype $DTYPE \
    --max-model-len $MAXLEN --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens $BT --max-num-seqs $NS --generation-config vllm'
EOF
start_server() {
  setsid bash "$OUT/serve.sh" >>"$OUT/server.log" 2>&1 &
  SERVER=$!
  local t0; t0=$(date +%s)
  until curl -sf "$URL/health" >/dev/null; do
    if ! kill -0 "$SERVER" 2>/dev/null; then echo "server died during startup"; tail -30 "$OUT/server.log"; return 1; fi
    if (( $(date +%s) - t0 > 1200 )); then echo "server not ready after 20 min"; return 1; fi
    sleep 3
  done
  echo "server ready after $(( $(date +%s) - t0 ))s"
}
stop_server() {
  [ -n "$SERVER" ] || return 0
  kill -INT -- "-$SERVER" 2>/dev/null; sleep 8; kill -KILL -- "-$SERVER" 2>/dev/null
  wait "$SERVER" 2>/dev/null; SERVER=""
}
trap 'stop_server' EXIT
trap 'stop_server; exit 143' TERM INT

metric() { curl -s "$URL/metrics" | awk -v m="$1" '$1 ~ "^"m"\\{" {s+=$2} END {print s+0}'; }
wait_idle() {
  for _ in $(seq 1 900); do
    r=$(metric vllm:num_requests_running); w=$(metric vllm:num_requests_waiting)
    if [ "${r%.*}" = 0 ] && [ "${w%.*}" = 0 ]; then return 0; fi
    sleep 1
  done
  echo "engine not idle after 15 min"
}
replay() {  # replay <name>: replays <name>.csv and records its wall-clock window
  local run=$1 start end
  start=$(python3 -c 'import time; print(time.time())')
  in_image python3 "$REPO/bench/replay.py" --url "$URL" --model m --workload "$OUT/$run.csv" \
    --out "$OUT/$run.jsonl" 2>>"$OUT/replay.log" || echo "replay $run failed"
  wait_idle
  end=$(python3 -c 'import time; print(time.time())')
  python3 - "$OUT/$run.jsonl" <<'PY' && echo "{\"run\": \"$run\", \"start\": $start, \"end\": $end}" >>"$OUT/runs.jsonl"
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if '"summary"' not in l]
bad = sum(1 for r in recs if r.get("error") or len(r.get("tok", ())) != r["output_len"])
print(f"  {sys.argv[1].rsplit('/', 1)[-1]}: {len(recs) - bad}/{len(recs)} requests complete")
sys.exit(1 if not recs or bad > 0.01 * len(recs) else 0)
PY
  sleep 2
}

start_server || exit 1
jt profile --run-dir "$OUT" --model "$MODEL_KEY" || exit 1
replay smoke
replay profile
jt plan --run-dir "$OUT" --model "$MODEL_KEY" --traces "$ROOT/data" || { echo "plan failed"; exit 1; }
for run in $(python3 -c "import json; print(' '.join(m['name'] for m in json.load(open('$OUT/validation_plan.json'))))"); do
  if ! curl -sf "$URL/health" >/dev/null; then echo "server down before $run"; break; fi
  replay "$run"
done
curl -s "$URL/metrics" >"$OUT/metrics_final.txt"
echo "done: $OUT"
