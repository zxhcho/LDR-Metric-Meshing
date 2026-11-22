import argparse
import os

from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mesh", type=str,
                      help="init computational domain directory")
    args.add_argument("--obs", type=str, help="dense obstacle field directory")
    args.add_argument("--init", type=str, help="initial solution directory")
    args.add_argument("--gt", type=str, help="GT solution directory")
    args.add_argument("--ne_it", type=int, default=100,
                      help="nonlinear Newton iteration")
    args.add_argument("--amr_it", type=int, default=10, help="AMR iteration")
    args.add_argument("--tol", type=float, default=1e-3,
                      help="convergence tolerance, relative L1 norm")
    args.add_argument("--stol", type=float, default=1e-4,
                      help="convergence tolerance, step norm")
    args.add_argument("--n", type=int, default=12)

    args = args.parse_args()

    poisson_binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "ex36_mesh_AMR")

    mesh_dir = args.mesh
    obs_dir = args.obs
    init_dir = args.init
    gt_dir = args.gt
    adapt_iter = args.amr_it
    newton_iter = args.ne_it
    l1_tolerance = args.tol
    step_tolerance = args.stol
    solution_base_dir = f"{mesh_dir}_AMR_{adapt_iter}_iter_{newton_iter}_tol_{l1_tolerance}_{step_tolerance}"
    solution_dir = os.path.join(solution_base_dir, "solution")
    log_dir = os.path.join(solution_base_dir, "log")

    os.makedirs(solution_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    filename_list = sorted(os.listdir(obs_dir))

    def solve(i):
        fname = filename_list[i]
        mesh_file = os.path.join(mesh_dir, fname)
        obs_file = os.path.join(obs_dir, fname)
        gt_file = os.path.join(gt_dir, fname)
        result_file = os.path.join(solution_dir, fname)
        log_file = os.path.join(log_dir, fname.replace(".vtk", ".yaml"))

        if init_dir is None:
            poisson_cmd = (
                f"{poisson_binary} "
                f"--mesh {mesh_file} "
                f"--obs {obs_file} "
                f"--gt {gt_file} "
                f"--result {result_file} "
                f"--max-amr {adapt_iter} "
                f"--max-it {newton_iter} "
                f"--tol {l1_tolerance} "
                f"--step-tol {step_tolerance} "
                f"--log {log_file} "
            )
        else:
            init_file = os.path.join(init_dir, fname)
            poisson_cmd = (
                f"{poisson_binary} "
                f"--mesh {mesh_file} "
                f"--obs {obs_file} "
                f"--init {init_file} "
                f"--gt {gt_file} "
                f"--result {result_file} "
                f"--max-amr {adapt_iter} "
                f"--max-it {newton_iter} "
                f"--tol {l1_tolerance} "
                f"--step-tol {step_tolerance} "
                f"--log {log_file} "
            )

        os.system(poisson_cmd)

    process_map(solve, range(len(filename_list)),
                max_workers=args.n, chunksize=1)
