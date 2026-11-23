#!/usr/bin/env bash
# Make the step probe discoverable by vLLM without pip: put this directory on PYTHONPATH.
#   bench/install_probe.sh <dir>
set -euo pipefail
DIR=${1:?usage: install_probe.sh <dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DIR/batchsim_vllm_probe-0.1.dist-info"
cp "$HERE/batchsim_vllm_probe.py" "$DIR/"
cat >"$DIR/batchsim_vllm_probe-0.1.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: batchsim-vllm-probe
Version: 0.1
EOF
cat >"$DIR/batchsim_vllm_probe-0.1.dist-info/entry_points.txt" <<'EOF'
[vllm.general_plugins]
batchsim_probe = batchsim_vllm_probe:register
EOF
echo "probe installed in $DIR"
