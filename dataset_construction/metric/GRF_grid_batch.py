import argparse
import time
import os

import numpy as np

from metric.GRF import GRF_grid
from geometric_kernels.spaces import Mesh
from metric import vtk_io

from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mesh", type=str, help="input obj mesh")
    args.add_argument("--batch", type=int, default=100)
    args.add_argument("--scale_min", type=float, default=0.05)
    args.add_argument("--scale_max", type=float, default=0.10)
    args.add_argument("--grid", type=int, default=100)
    args.add_argument("--n", type=int, default=1)
    args = args.parse_args()

    mesh_path = args.mesh
    mesh = Mesh.load_mesh(mesh_path)

    batch_num = args.batch
    seed = int(time.time())
    np.random.seed(seed)
    scale_min = args.scale_min
    scale_max = args.scale_max
    scale_batch = np.random.uniform(scale_min, scale_max, size=batch_num)

    field_dir = mesh_path.replace(
        ".obj", f"_field_{scale_min}_{scale_max}_{batch_num}")
    field_dir = os.path.join(field_dir, "rhs")
    os.makedirs(field_dir, exist_ok=True)

    def compute(i):
        seed = np.abs(np.array(time.time_ns()).astype(np.int32))
        mesh.vertices[:, 0] += 0.5
        mesh.vertices[:, 1] += 0.5
        scale = scale_batch[i]
        u = GRF_grid(mesh.vertices[:, :2], args.grid, scale, 1, seed)
        mesh.vertices[:, 0] -= 0.5
        mesh.vertices[:, 1] -= 0.5
        vtk_io.write_vtk_with_attribute(
            "{}/{}_{:.2}.vtk".format(field_dir, str(i).zfill(4), scale), mesh.vertices, mesh.faces, u.reshape(-1))

    process_map(compute, range(batch_num), max_workers=args.n, chunksize=1)
