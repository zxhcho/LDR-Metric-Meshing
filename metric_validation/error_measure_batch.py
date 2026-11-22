import os
import argparse

from tqdm.contrib.concurrent import process_map


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--gt", type=str)
    args.add_argument("--solve", type=str)
    args.add_argument("--order", type=int, default=1)
    args.add_argument("--n", type=int, default=12)
    args = args.parse_args()

    error_binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "discrete_error")

    gt_dir = args.gt
    solve_dir = args.solve
    order = args.order
    log_dir = solve_dir + f"_d_log_{order}"
    error_dir = solve_dir + f"_d_error_{order}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    fname_list = os.listdir(solve_dir)

    def measure(i):
        fname = fname_list[i]
        gt_file = os.path.join(gt_dir, fname)
        solve_file = os.path.join(solve_dir, fname)
        log_file = os.path.join(log_dir, fname.replace(".vtk", ".yaml"))
        error_file = os.path.join(error_dir, fname)
        cmd = f"{error_binary} --gt={gt_file} --solve={solve_file} --log={log_file} --error={error_file} --order={order}"
        os.system(cmd)

    process_map(measure, range(len(fname_list)),
                max_workers=args.n, chunksize=1)
