import argparse
import os

from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--field", type=str)
    args.add_argument("--k", type=int, default=3)
    args.add_argument("--dim", type=int, default=2)
    args.add_argument("--n", type=int, default=12)

    args = args.parse_args()

    poisson_binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "field_metric_ref")

    k = args.k
    field_dir = args.field
    metric_dir = f"{field_dir}_metric_{k}"
    os.makedirs(metric_dir, exist_ok=True)

    fname_list = sorted(os.listdir(field_dir))

    def solve(i):
        fname = fname_list[i]
        field_file = os.path.join(field_dir, fname)
        metric_file = os.path.join(metric_dir, fname.replace(".vtk", ".mat"))

        poisson_cmd = (
            f"{poisson_binary} "
            f"--field={field_file} "
            f"--metric={metric_file} "
            f"--dim={args.dim} "
            f"--k={k} "
        )
        os.system(poisson_cmd)

    process_map(solve, range(len(fname_list)),
                max_workers=args.n, chunksize=1)
