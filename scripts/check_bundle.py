"""Check that the packaged forecasting bundle is exactly what ``results/`` fits to.

python scripts/check_bundle.py            # exit 1 on any mismatch
python scripts/check_bundle.py --write    # refit and overwrite the packaged bundle
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from gpufc.bundle import fit_bundle
from gpufc.cli import default_bundle
from gpufc.serving import Bundle


def diff(a: Any, b: Any, path: str = "", rtol: float = 1e-6) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'packaged' if k in a else 'refit'}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}", rtol)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in diff(x, y, f"{path}[{i}]", rtol)]
    if isinstance(a, float) or isinstance(b, float):
        if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12):
            return [f"{path}: {a!r} != {b!r}"]
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    refit = fit_bundle(Path(args.results))
    if args.write:
        refit.save(default_bundle())
        print(f"wrote {default_bundle()}")
        return 0
    packaged = Bundle.load(default_bundle())
    problems = diff(packaged.to_dict(), refit.to_dict())
    for p in problems[:20]:
        print(p)
    print("bundle reproduces" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
