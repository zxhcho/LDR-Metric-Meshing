import os
import argparse

from tqdm.contrib.concurrent import process_map


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--obj", type=str)
    args.add_argument("--field", type=str)
    args.add_argument("--project", type=str)
    args.add_argument("--n", type=int, default=12)
    args = args.parse_args()

    project_binary = os.path.join(os.environ["METRIC_BIN_ROOT"], "field_projection")

    field_dir = args.field
    obj_dir = args.obj
    project_dir = args.project

    os.makedirs(project_dir, exist_ok=True)
    fname_list = sorted(os.listdir(field_dir))

    def project(i):
        fname = fname_list[i]
        field_path = os.path.join(field_dir, fname)
        project_path = os.path.join(project_dir, fname)
        if os.path.isdir(obj_dir):
            obj_path = os.path.join(obj_dir, fname.replace(".vtk", ".obj"))
        else:
            obj_path = obj_dir
        cmd = (
            f"{project_binary} "
            f"--obj={obj_path} "
            f"--field={field_path} "
            f"--project={project_path} "
        )
        os.system(cmd)

    process_map(project, range(len(fname_list)),
                max_workers=args.n, chunksize=1)
