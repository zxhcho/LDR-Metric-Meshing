"""Compare metric scalars in pairs of VTK files.

Each file must contain one SCALARS block named `metric`. The comparison is
the max absolute difference of those scalars, in file-name order. Paths are
arguments. This script does not read environment variables or host paths.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def metric_scalars(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    marker = "SCALARS metric double 1\n"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"no metric scalar block: {path}")
    body = text[start + len(marker):]
    lines = [line.strip() for line in body.splitlines()]
    if not lines or not lines[0].startswith("LOOKUP_TABLE"):
        raise SystemExit(f"missing LOOKUP_TABLE after metric scalars: {path}")
    values = [float(line) for line in lines[1:] if line]
    if not values:
        raise SystemExit(f"empty metric scalar block: {path}")
    return np.asarray(values, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    predicted = sorted(path for path in args.prediction.glob("*.vtk") if path.is_file())
    reference = {path.name: path for path in args.reference.glob("*.vtk") if path.is_file()}
    if not predicted:
        raise SystemExit(f"no vtk files in {args.prediction}")
    missing = [path.name for path in predicted if path.name not in reference]
    if missing:
        raise SystemExit(f"reference missing {len(missing)} files, first={missing[0]}")
    worst_name = None
    worst = 0.0
    for path in predicted:
        pred = metric_scalars(path)
        ref = metric_scalars(reference[path.name])
        if pred.shape != ref.shape:
            raise SystemExit(
                f"length mismatch {path.name}: {pred.shape[0]} vs {ref.shape[0]}")
        diff = float(np.max(np.abs(pred - ref)))
        if diff >= worst:
            worst = diff
            worst_name = path.name
    print(f"files={len(predicted)} max_abs={worst:.16e} worst={worst_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
