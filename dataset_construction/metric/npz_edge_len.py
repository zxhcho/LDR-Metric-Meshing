import os
import numpy as np
import argparse
from tqdm.contrib.concurrent import process_map

from vtk_io import read_vtk_with_attribute

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--field", type=str,
                      help="Right hand size field for equation")
    args.add_argument("--edge", type=str,
                      help="Metric induced edge length")
    args.add_argument("--npz", type=str)
    args.add_argument("--n", type=int, default=12)
    args = args.parse_args()

    edge_dir = args.edge
    field_dir = args.field
    npz_dir = args.npz

    fname_list = sorted(os.listdir(edge_dir))

    def save_npz(i):
        fname = fname_list[i]
        edge_path = os.path.join(edge_dir, fname)
        field_path = os.path.join(field_dir, fname)
        npz_path = os.path.join(npz_dir, fname.replace(".vtk", ".npz"))
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)

        _, edges, length = read_vtk_with_attribute(edge_path)
        _, _, field = read_vtk_with_attribute(field_path)
        ne = edges.shape[0]

        # for half edge case
        if (edges[0, :] == edges[ne//2, :]).all():
            ne = ne//2
            edges = edges[:ne, :]
            length = np.stack(
                [length[:ne], length[ne:]], axis=1)
        else:
            length = length.reshape(-1, 1)

        np.savez_compressed(npz_path,
                            edges=edges,
                            length=length,
                            field=field,
                            )

    process_map(save_npz, range(len(fname_list)),
                max_workers=args.n, chunksize=1)
