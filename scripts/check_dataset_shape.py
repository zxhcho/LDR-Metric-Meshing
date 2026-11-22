"""Count mesh records and log samples for the obstacle release."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_layout import LOGS, MESH, REFERENCE_EDGE


def count_obj(path):
    vertices = 0
    faces = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return vertices, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    obj = args.data_root / MESH
    vertices, faces = count_obj(obj)
    print(f"mesh/plane_44k.obj vertices={vertices} faces={faces}")
    failed = vertices != 44304 or faces != 87672
    for relative in LOGS:
        with (args.data_root / relative).open() as handle:
            data = yaml.safe_load(handle)
        n = len(data)
        print(f"samples={n} {relative}")
        if n != 100:
            failed = True
    reference = args.data_root / REFERENCE_EDGE
    files = [path for path in reference.iterdir() if path.is_file()]
    vtk = [path for path in files if path.name.endswith(".vtk")]
    print(f"reference_vtk files={len(files)} vtk={len(vtk)}")
    if len(files) != 100 or len(vtk) != 100:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
