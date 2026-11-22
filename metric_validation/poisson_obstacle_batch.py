import argparse
import os

from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mesh", type=str, help="mesh field directory")
    args.add_argument("--init", type=str, help="initial solution directory")
    args.add_argument("--gt", type=str, help="GT solution directory")
    args.add_argument("--iter", type=int, default=100)
    args.add_argument("--tol", type=float, default=1e-3)
    args.add_argument("--stol", type=float, default=1e-4)
    args.add_argument("--n", type=int, default=12)

    args = args.parse_args()

    poisson_binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "ex36_mesh")

    mesh_dir = args.mesh
    init_dir = args.init
    gt_dir = args.gt
    num_iter = args.iter
    tolerance = args.tol
    step_tolerance = args.stol
    solution_base_dir = f"{mesh_dir}_iter_{num_iter}_tol_{tolerance}_{step_tolerance}"
    solution_dir = os.path.join(solution_base_dir, "solution")
    log_dir = os.path.join(solution_base_dir, "log")

    os.makedirs(solution_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    filename_list = sorted(os.listdir(mesh_dir))

    def solve(i):
        fname = filename_list[i]
        mesh_file = os.path.join(mesh_dir, fname)
        solution_file = os.path.join(solution_dir, fname)
        gt_file = os.path.join(gt_dir, fname)
        log_file = os.path.join(log_dir, fname.replace(".vtk", ".yaml"))

        if init_dir is None:
            poisson_cmd = f"{poisson_binary} -m {mesh_file} -r {solution_file} --gt {gt_file} -mi {num_iter} -tol {tolerance} -stol {step_tolerance} -l {log_file}"
        else:
            init_file = os.path.join(init_dir, fname)
            poisson_cmd = f"{poisson_binary} -m {mesh_file} -r {solution_file} --init {init_file} --gt {gt_file} -mi {num_iter} -tol {tolerance} -stol {step_tolerance} -l {log_file} "

        os.system(poisson_cmd)

    process_map(solve, range(len(filename_list)),
                max_workers=args.n, chunksize=1)
