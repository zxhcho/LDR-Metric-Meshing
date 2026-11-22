"""Recompute Table 3 mean and sample standard deviation from seven logs.

Each log is a mapping of sample id to a record with total_time, DOF, and
final_rel_l1. The sample standard deviation uses ddof=1. Time is rounded to
2 decimal places. DOF is rounded to an integer. final_rel_l1 is multiplied by
1000 and then rounded to 2 decimal places. The printed cells are EXPECTED.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_layout import LOGS

# (label, time_mean, time_std, metric, metric_mean, metric_std)
# metric is "dof" or "err". err is final_rel_l1 * 1e3 at the printed precision.
EXPECTED = [
    ("Ours AMR", 5.93, 1.95, "dof", 6653, 2227),
    ("Ours", 4.14, 1.05, "err", 0.99, 0.18),
    ("Obs AMR", 10.99, 1.89, "dof", 8756, 1487),
    ("Obs", 5.53, 1.51, "err", 1.65, 0.31),
    ("Uni 4k AMR", 15.05, 4.30, "dof", 9069, 6454),
    ("Uni 4k", 4.65, 1.67, "err", 3.08, 0.64),
    ("Uni 12k", 25.87, 7.04, "err", 1.11, 0.36),
]


def load_columns(path):
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a mapping")
    times, dofs, errors = [], [], []
    for sample in data.values():
        times.append(float(sample["total_time"]))
        dofs.append(float(sample["DOF"]))
        errors.append(float(sample["final_rel_l1"]))
    return np.asarray(times), np.asarray(dofs), np.asarray(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    failed = False
    for relative, expected in zip(LOGS, EXPECTED):
        label, time_mean, time_std, kind, metric_mean, metric_std = expected
        path = args.data_root / relative
        times, dofs, errors = load_columns(path)
        n = times.shape[0]
        got_time = (round(float(times.mean()), 2), round(float(times.std(ddof=1)), 2))
        got_dof = (int(round(float(dofs.mean()))), int(round(float(dofs.std(ddof=1)))))
        got_err = (
            round(float(errors.mean()) * 1000.0, 2),
            round(float(errors.std(ddof=1)) * 1000.0, 2),
        )
        if kind == "dof":
            got_metric = got_dof
        else:
            got_metric = got_err
        ok = n == 100 and got_time == (time_mean, time_std) and got_metric == (metric_mean, metric_std)
        status = "ok" if ok else "MISMATCH"
        print(
            f"{status} {label} n={n} "
            f"time=({got_time[0]:.2f}, {got_time[1]:.2f}) "
            f"dof=({got_dof[0]}, {got_dof[1]}) "
            f"err_x1e3=({got_err[0]:.2f}, {got_err[1]:.2f}) "
            f"expected_time=({time_mean:.2f}, {time_std:.2f}) "
            f"expected_{kind}=({metric_mean}, {metric_std})"
        )
        if not ok:
            failed = True
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
