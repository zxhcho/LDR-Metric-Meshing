import os
import argparse
import yaml

from tqdm.contrib.concurrent import process_map


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--log", type=str)
    args = args.parse_args()

    log_dir = args.log
    merge_file = log_dir + "_merge.yaml"

    fnames = os.listdir(log_dir)
    fnames = sorted(fnames)

    data_merge = {}
    for i in range(len(fnames)):
        fname = fnames[i]
        fidx = str(fname.replace(".yaml", ""))
        log_file = os.path.join(log_dir, fname)
        with open(log_file, 'r') as f:
            data = yaml.safe_load(f)
            data_merge[fidx] = data

    with open(merge_file, 'w') as f:
        yaml.dump(data_merge, f)
