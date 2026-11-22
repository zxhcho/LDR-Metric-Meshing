import argparse
import os
import igl

from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--obj", type=str, help="obj file")
    args.add_argument("--metric", type=str,
                      help="directory with metric file")
    args.add_argument("--dof", type=int, default=-1)
    args.add_argument("--hmin", type=float, default=1)
    args.add_argument("--hmax", type=float, default=10)
    args.add_argument("--iso", type=int, default=0)
    args.add_argument("--n", type=int, default=12)

    args = args.parse_args()

    MMG_2D_binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "mmg_2d_metric")

    metric_dir = args.metric
    dof = args.dof
    hmin = args.hmin
    hmax = args.hmax
    is_isotropic = args.iso
    remesh_dir = metric_dir + f"_MMG_2D_{dof}_{hmin}_{hmax}"
    if is_isotropic:
        remesh_dir += "_iso"

    os.makedirs(remesh_dir, exist_ok=True)
    filename_list = sorted(os.listdir(metric_dir))

    def solve(i):
        fname = filename_list[i]
        if os.path.isdir(args.obj):
            obj_file = os.path.join(args.obj, fname.replace(".mat", ".obj"))
        else:
            obj_file = args.obj
        metric_file = os.path.join(metric_dir, fname)
        remesh_file = os.path.join(remesh_dir, fname.replace(".mat", ".obj"))

        MMG_cmd = f"{MMG_2D_binary} --in={obj_file} --metric={metric_file} --out={remesh_file} --dof={args.dof} --hmin={hmin} --hmax={hmax}"
        if is_isotropic:
            MMG_cmd += " --isotropic=1"
        os.system(MMG_cmd)

    process_map(solve, range(len(filename_list)),
                max_workers=args.n, chunksize=1)
