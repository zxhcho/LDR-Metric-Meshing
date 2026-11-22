"""Reproduce table tab:obstacle_convergence_tab.

Chain, with the parameters encoded in the published directory names:

1. Network inference writes edge-length VTK files.
2. metric_from_edge recovers a metric, dim=2, dual=0.
3. mmg_2d_metric remeshes at dof=4000, hmin=0.1, hmax=10.
4. field_projection maps field/obstacle and field/init onto that mesh.
   field/init is the stored 6k solution used as the initial guess.
5. ex36_mesh solves with max-it=100, tol=0.001, step-tol=1e-5.
   ex36_mesh_AMR solves with max-amr=100, max-it=1, and the same tolerances.
   Each solver process is pinned to one thread.
6. Per-sample logs are merged to log_merge.yaml under --work-root.

Ours starts from checkpoint/model_100.pth. The relative L1 ground truth is
field/gt. Obs and the two uniform arms read --baseline-root, which is not
inside the release archive:

- BASE/obs_metric/*.mat
- BASE/uni4k_mesh/*.vtk
- BASE/uni4k_init/*.vtk
- BASE/uni12k_mesh/*.vtk
- BASE/uni12k_init/*.vtk

Without --baseline-root the script runs the two Ours arms and prints that
the baseline arms were skipped.

Outputs are written under --work-root. Inputs are symlinked from --data-root.
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_layout import (
    FIELD_GT,
    FIELD_INIT,
    FIELD_OBSTACLE,
    HASH,
    link_path,
    link_tree,
    materialize_infer_tree,
)

MMG_TAG = "MMG_2D_4000_0.1_10"
SOLVE_TAG = "iter_100_tol_0.001_1e-05"
AMR_TAG = "AMR_100_iter_1_tol_0.001_1e-05"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def run_cmd(cmd):
    env = os.environ.copy()
    env.update(THREAD_ENV)
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def parallel(commands, workers):
    if not commands:
        return
    if workers == 1 or len(commands) == 1:
        for cmd in commands:
            run_cmd(cmd)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_cmd, commands))


def vtk_names(directory: Path):
    return sorted(path.name for path in directory.glob("*.vtk"))


def mat_names(directory: Path):
    return sorted(path.name for path in directory.glob("*.mat"))


def obj_names(directory: Path):
    return sorted(path.name for path in directory.glob("*.obj"))


def require_count(directory: Path, suffix: str, expected: int, label: str):
    found = len(list(directory.glob(f"*{suffix}")))
    if found != expected:
        raise SystemExit(f"{label}: {found} {suffix} files, expected {expected}")


def prepare(data_root: Path, work: Path, only: str, baseline_root: Path):
    materialize_infer_tree(data_root, work, only)
    link_tree(data_root / FIELD_OBSTACLE, work / FIELD_OBSTACLE, only)
    link_tree(data_root / FIELD_INIT, work / FIELD_INIT, only)
    link_tree(data_root / FIELD_GT, work / FIELD_GT, only)
    if baseline_root is None:
        return False
    link_tree(baseline_root / "obs_metric", work / "baseline/obs_metric", only)
    for name in ("uni4k_mesh", "uni4k_init", "uni12k_mesh", "uni12k_init"):
        link_tree(baseline_root / name, work / "baseline" / name, only)
    return True


def infer(repo: Path, work: Path, python: str):
    env = os.environ.copy()
    env["METRIC_DATA_ROOT"] = str(work)
    env["PYTHONPATH"] = str(repo / "metric_learning")
    env.update(THREAD_ENV)
    cmd = [
        python, "-m", "pipeline.run",
        "--config", str(repo / "metric_learning/config.example.yaml"),
        "--mode", "infer",
    ]
    completed = subprocess.run(cmd, env=env, cwd=repo / "metric_learning")
    if completed.returncode != 0:
        raise SystemExit(f"inference failed ({completed.returncode})")


def metric_commands(binary: Path, obj: Path, edge_dir: Path, metric_dir: Path):
    metric_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name in vtk_names(edge_dir):
        out = metric_dir / name.replace(".vtk", ".mat")
        if out.is_file() and out.stat().st_size > 0:
            continue
        commands.append([
            str(binary), f"--obj={obj}", f"--edge={edge_dir / name}",
            f"--metric={out}", "--dim=2", "--dual=0",
        ])
    return commands


def mmg_commands(binary: Path, obj: Path, metric_dir: Path, mesh_dir: Path):
    mesh_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name in mat_names(metric_dir):
        out = mesh_dir / name.replace(".mat", ".obj")
        if out.is_file() and out.stat().st_size > 0:
            continue
        commands.append([
            str(binary), f"--in={obj}", f"--metric={metric_dir / name}",
            f"--out={out}", "--dof=4000", "--hmin=0.1", "--hmax=10",
        ])
    return commands


def project_commands(binary: Path, mesh_dir: Path, field_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name in obj_names(mesh_dir):
        vtk_name = name.replace(".obj", ".vtk")
        out = out_dir / vtk_name
        if out.is_file() and out.stat().st_size > 0:
            continue
        commands.append([
            str(binary), f"--obj={mesh_dir / name}",
            f"--field={field_dir / vtk_name}", f"--project={out}",
        ])
    return commands


def solve_commands(binary: Path, mesh_dir: Path, init_dir: Path, gt_dir: Path, out_root: Path):
    solution_dir = out_root / "solution"
    log_dir = out_root / "log"
    solution_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name in vtk_names(mesh_dir):
        log_file = log_dir / name.replace(".vtk", ".yaml")
        if log_file.is_file() and log_file.stat().st_size > 0:
            continue
        commands.append([
            str(binary),
            "-m", str(mesh_dir / name),
            "-r", str(solution_dir / name),
            "--init", str(init_dir / name),
            "--gt", str(gt_dir / name),
            "-mi", "100",
            "-tol", "0.001",
            "-stol", "1e-5",
            "-l", str(log_file),
        ])
    return commands


def amr_commands(binary, mesh_dir, obs_dir, init_dir, gt_dir, out_root):
    solution_dir = out_root / "solution"
    log_dir = out_root / "log"
    solution_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name in vtk_names(mesh_dir):
        log_file = log_dir / name.replace(".vtk", ".yaml")
        if log_file.is_file() and log_file.stat().st_size > 0:
            continue
        commands.append([
            str(binary),
            "--mesh", str(mesh_dir / name),
            "--obs", str(obs_dir / name),
            "--init", str(init_dir / name),
            "--gt", str(gt_dir / name),
            "--result", str(solution_dir / name),
            "--max-amr", "100",
            "--max-it", "1",
            "--tol", "0.001",
            "--step-tol", "1e-5",
            "--log", str(log_file),
        ])
    return commands


def merge_logs(log_dir: Path):
    import yaml
    merged = {}
    for path in sorted(log_dir.glob("*.yaml")):
        merged[path.stem] = yaml.safe_load(path.read_text())
    out = Path(str(log_dir) + "_merge.yaml")
    out.write_text(yaml.safe_dump(merged), encoding="utf-8")
    print(f"merged {len(merged)} {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--bin-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", default="", help="sample prefix, empty runs every file")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="directory with obs_metric, uni4k_mesh, uni4k_init, uni12k_mesh, uni12k_init",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    data_root = args.data_root.resolve()
    work = args.work_root.resolve()
    bin_root = args.bin_root.resolve()
    for name in (
        "metric_from_edge", "mmg_2d_metric", "field_projection",
        "ex36_mesh", "ex36_mesh_AMR",
    ):
        if not (bin_root / name).is_file():
            raise SystemExit(f"missing binary {bin_root / name}")

    work.mkdir(parents=True, exist_ok=True)
    baseline_root = None if args.baseline_root is None else args.baseline_root.resolve()
    has_baseline = prepare(data_root, work, args.only, baseline_root)
    n_predict = len(list((work / "model/metric_3/predict").glob("*.npz")))
    print(f"samples={n_predict}")

    edge_dir = work / "model/metric_3/predict_out" / HASH / "model_100.pth"
    if not args.skip_infer:
        infer(repo, work, args.python)
    require_count(edge_dir, ".vtk", n_predict, "inference")

    metric_dir = Path(str(edge_dir) + "_metric")
    parallel(
        metric_commands(bin_root / "metric_from_edge", work / "mesh/plane_44k.obj", edge_dir, metric_dir),
        args.workers,
    )
    require_count(metric_dir, ".mat", n_predict, "metric")

    background = work / "mesh/plane_44k.obj"
    obs_field = work / FIELD_OBSTACLE
    init_field = work / FIELD_INIT
    gt_dir = work / FIELD_GT

    def project_and_solve(metric_dir_for_arm):
        mesh_dir = Path(str(metric_dir_for_arm) + "_" + MMG_TAG)
        obs_mesh = Path(str(mesh_dir) + "_obs")
        init_mesh = Path(str(mesh_dir) + "_init")
        parallel(mmg_commands(bin_root / "mmg_2d_metric", background, metric_dir_for_arm, mesh_dir), args.workers)
        require_count(mesh_dir, ".obj", n_predict, mesh_dir.name)
        parallel(project_commands(bin_root / "field_projection", mesh_dir, obs_field, obs_mesh), args.workers)
        parallel(project_commands(bin_root / "field_projection", mesh_dir, init_field, init_mesh), args.workers)
        require_count(obs_mesh, ".vtk", n_predict, obs_mesh.name)
        require_count(init_mesh, ".vtk", n_predict, init_mesh.name)
        solve_root = Path(str(obs_mesh) + "_" + SOLVE_TAG)
        amr_root = Path(str(obs_mesh) + "_" + AMR_TAG)
        parallel(solve_commands(bin_root / "ex36_mesh", obs_mesh, init_mesh, gt_dir, solve_root), args.workers)
        parallel(amr_commands(bin_root / "ex36_mesh_AMR", obs_mesh, obs_field, init_mesh, gt_dir, amr_root), args.workers)
        require_count(solve_root / "log", ".yaml", n_predict, solve_root.name)
        require_count(amr_root / "log", ".yaml", n_predict, amr_root.name)
        merge_logs(solve_root / "log")
        merge_logs(amr_root / "log")

    project_and_solve(metric_dir)
    if not has_baseline:
        print("baseline arms skipped: --baseline-root was not set")
        print(f"work-root={work}")
        return 0
    project_and_solve(work / "baseline/obs_metric")

    iso = work / "baseline"
    for mesh_name, init_name, with_amr in (
        ("uni4k_mesh", "uni4k_init", True),
        ("uni12k_mesh", "uni12k_init", False),
    ):
        mesh_dir = iso / mesh_name
        init_dir = iso / init_name
        selected = vtk_names(mesh_dir)
        if args.only:
            selected = [name for name in selected if name.startswith(args.only)]
        # Uniform arms use the stored meshes. A prefix run still solves the
        # selected names by pointing the solvers at a filtered directory.
        if args.only and len(selected) != n_predict:
            filtered = work / "filtered" / mesh_name
            filtered_init = work / "filtered" / init_name
            if filtered.exists():
                shutil.rmtree(filtered)
            if filtered_init.exists():
                shutil.rmtree(filtered_init)
            filtered.mkdir(parents=True)
            filtered_init.mkdir(parents=True)
            for name in selected:
                link_path(mesh_dir / name, filtered / name)
                link_path(init_dir / name, filtered_init / name)
            mesh_dir = filtered
            init_dir = filtered_init
        solve_root = Path(str(mesh_dir) + "_" + SOLVE_TAG)
        parallel(solve_commands(bin_root / "ex36_mesh", mesh_dir, init_dir, gt_dir, solve_root), args.workers)
        require_count(solve_root / "log", ".yaml", len(vtk_names(mesh_dir)), solve_root.name)
        merge_logs(solve_root / "log")
        if with_amr:
            amr_root = Path(str(mesh_dir) + "_" + AMR_TAG)
            parallel(
                amr_commands(bin_root / "ex36_mesh_AMR", mesh_dir, obs_field, init_dir, gt_dir, amr_root),
                args.workers,
            )
            require_count(amr_root / "log", ".yaml", len(vtk_names(mesh_dir)), amr_root.name)
            merge_logs(amr_root / "log")

    print(f"work-root={work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
