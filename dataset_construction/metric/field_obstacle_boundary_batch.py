import os
import argparse

from metric.field_operation import field_obstacle_boundary
from tqdm.contrib.concurrent import process_map


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--field", type=str)
    args.add_argument("--n", type=int, default=12)
    args = args.parse_args()

    field_dir = args.field
    visual_dir = field_dir + f"_obs_boundary"

    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(field_dir, exist_ok=True)
    fname_list = sorted(os.listdir(field_dir))

    def visualize(i):
        fname = fname_list[i]
        field_path = os.path.join(field_dir, fname)
        visual_path = os.path.join(visual_dir, fname)
        field_obstacle_boundary(field_path, visual_path)

    process_map(visualize, range(len(fname_list)),
                max_workers=args.n, chunksize=1)
