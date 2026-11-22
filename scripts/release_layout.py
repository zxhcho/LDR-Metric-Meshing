"""Flat paths inside obstacle-40k_obs-predict.tar, and the symlink tree inference expects.

pipeline.run reads three paths under METRIC_DATA_ROOT:

- model/metric_3/predict/*.npz
- model/metric_3/op_cache/<one npz>
- model/metric_3/saved_models/<hash>/model_100.pth

materialize_infer_tree() builds those three paths as symlinks into --work-root.
The archive itself keeps the short paths below. Inference writes VTK under the
work tree, so it does not replace reference/edge.
"""

import argparse
import sys
from pathlib import Path

HASH = (
    "DiscreteEdge_8_128_field_Cri_DiscreteMetricAbs_1.0_Epo_100_"
    "Opt_Adam_0.0001_Sche_20_0.5"
)
CHECKPOINT = "checkpoint/model_100.pth"
CHECKPOINT_BYTES = 4906601
CHECKPOINT_SHA256 = (
    "cdebd292fb7a9fa394a486a8ff93eabe8d9db742d5db1ca07c351dac972783ff"
)
MESH = "mesh/plane_44k.obj"
INFER_SAMPLES = "infer/samples"
INFER_OP_CACHE = "infer/op_cache"
REFERENCE_EDGE = "reference/edge"
FIELD_OBSTACLE = "field/obstacle"
FIELD_INIT = "field/init"
FIELD_GT = "field/gt"
LOGS = [
    "log/ours_amr.yaml",
    "log/ours.yaml",
    "log/obs_amr.yaml",
    "log/obs.yaml",
    "log/uni4k_amr.yaml",
    "log/uni4k.yaml",
    "log/uni12k.yaml",
]

def link_path(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src)


def link_tree(src: Path, dst: Path, only: str):
    if not src.is_dir():
        raise SystemExit(f"missing directory: {src}")
    files = sorted(path for path in src.iterdir() if path.is_file())
    if only:
        files = [path for path in files if path.name.startswith(only)]
    if not files:
        raise SystemExit(f"no files selected in {src}")
    for path in files:
        link_path(path, dst / path.name)


def materialize_infer_tree(archive_root: Path, work: Path, only: str = ""):
    """Symlink the flat archive onto the paths pipeline.run opens."""
    link_tree(archive_root / INFER_SAMPLES, work / "model/metric_3/predict", only)
    link_tree(archive_root / INFER_OP_CACHE, work / "model/metric_3/op_cache", "")
    link_path(
        archive_root / CHECKPOINT,
        work / "model/metric_3/saved_models" / HASH / "model_100.pth",
    )
    link_path(archive_root / MESH, work / MESH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    archive_root = args.archive_root.resolve()
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    materialize_infer_tree(archive_root, work, args.only)
    print(f"work-root={work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
