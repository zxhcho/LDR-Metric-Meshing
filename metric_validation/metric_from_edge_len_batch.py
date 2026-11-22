import argparse
import os

from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--obj", type=str, help="shape file")
    args.add_argument("--edge", type=str, help="edge directory")
    args.add_argument("--n", type=int, default=12)
    args.add_argument("--dual", type=int, default=0)
    args.add_argument("--dim", type=int, default=2)

    args = args.parse_args()

    binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "metric_from_edge")

    edge_dir = args.edge
    metric_dir = edge_dir + "_metric"
    os.makedirs(metric_dir, exist_ok=True)

    filename_list = sorted(os.listdir(edge_dir))

    def calculate(i):
        fname = filename_list[i]
        if os.path.isdir(args.obj):
            obj_file = os.path.join(args.obj, fname.replace(".vtk", ".obj"))
        else:
            obj_file = args.obj
        edge_file = os.path.join(edge_dir, fname)
        fname = fname.replace(".vtk", ".mat")
        metric_file = os.path.join(metric_dir, fname)

        poisson_cmd = (
            f"{binary} "
            f"--obj={obj_file} "
            f"--edge={edge_file} "
            f"--metric={metric_file} "
            f"--dim={args.dim} "
            f"--dual={args.dual} "
        )
        os.system(poisson_cmd)

    process_map(calculate, range(len(filename_list)),
                max_workers=args.n, chunksize=1)
