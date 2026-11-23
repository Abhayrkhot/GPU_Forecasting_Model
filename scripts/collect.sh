#!/bin/bash
# Stream finished GPU runs from the cluster to the analysis machine and compact them.
#
#   EXPLORER=user@cluster-login scripts/collect.sh   # cluster -> $BOX:~/gpufc-runs -> results/
#
# EXPLORER is the cluster login (runs live in /scratch/$USER/gpufc/runs there),
# EXPLORER_KEY an optional ssh key for it, BOX the analysis host with this repository at
# REPO_ON_BOX. For every (gpu, model) the newest run directory that completed its profile
# replay and at least one validation replay is compacted into results/ (see gpufc.data).
set -euo pipefail
EXPLORER=${EXPLORER:?set EXPLORER=user@cluster-login}
BOX=${BOX:-box}
RAW=${RAW:-gpufc-runs}
REPO_ON_BOX=${REPO_ON_BOX:-Projects/gpufc}

ssh ${EXPLORER_KEY:+-i "$EXPLORER_KEY"} "$EXPLORER" \
  'cd /scratch/$USER/gpufc && tar cf - --exclude="*.csv" runs' \
  | ssh "$BOX" "mkdir -p $RAW && tar xf - -C $RAW"

ssh "$BOX" "cd $REPO_ON_BOX && .venv/bin/python - \$HOME/$RAW/runs" <<'PY'
import json, sys
from pathlib import Path
from gpufc.data import compact_run

root = Path(sys.argv[1]).expanduser()
best = {}
for d in sorted(root.glob("*/*/*")):
    runs = d / "runs.jsonl"
    if not runs.exists():
        continue
    names = [json.loads(x)["run"] for x in runs.read_text().splitlines() if x.strip()]
    if "profile" not in names or len(names) < 3:
        continue
    best[(d.parent.parent.name, d.parent.name)] = d  # sorted: newest job wins
for (g, m), d in sorted(best.items()):
    info, table = compact_run(d, Path("results"))
    print(f"{info.gpu:9s} {info.model:15s} {len(table):7d} steps {len(info.windows)} replays")
PY
