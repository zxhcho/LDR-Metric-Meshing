import argparse
import numpy as np

import vtk_io
from visualizer.build_visualizer import VISUALIZER_REGISTRY


def uv_to_3d(uv_coordinates, local_frames):
    """
        uv_coordinates: (n, 2)
        local_frames: (n, 6)
    """
    global_coords = local_frames[:, :3] * uv_coordinates[:, 0].reshape(-1, 1) +\
        local_frames[:, 3:] * uv_coordinates[:, 1].reshape(-1, 1)
    return global_coords


def visualize_ff(filename, bc, u, v, loss=None):
    nff = bc.shape[0]
    ff1 = bc + u
    ff2 = bc + v
    ff3 = bc - u
    ff4 = bc - v
    verts = np.concatenate([bc, ff1, ff2, ff3, ff4], axis=1).reshape(-1, 3)

    faces0 = np.arange(nff).reshape(-1, 1)*5
    faces = np.concatenate(
        [faces0, faces0+1,
         faces0, faces0+2,
         faces0, faces0+3,
         faces0, faces0+4], axis=1).reshape(4*nff, 2)

    if loss is None:
        vtk_io.write_vtk(filename, verts, faces)
    else:
        loss = loss.reshape(-1, 1)
        prop = np.concatenate([loss, loss, loss, loss], axis=1).reshape(-1)
        vtk_io.write_vtk_with_attribute(filename, verts, faces, prop, "loss")


@VISUALIZER_REGISTRY("FrameField")
def framefield_visualizer(filename: str, origins, local_frame, uv, loss=None):
    """
        origins: (n, 3), origin for local frame
        local_frame: (n, 6), local frame
        uv: (n, 4)
    """
    nv1 = uv[:, :2]
    nv2 = uv[:, 2:]
    ff1 = uv_to_3d(nv1, local_frame)
    ff2 = uv_to_3d(nv2, local_frame)
    visualize_ff(filename, origins, ff1, ff2, loss=loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str)
    parser.add_argument("--ff", type=str)
    args = parser.parse_args()

    data = np.load(args.npz)
    bc = data["bc"]
    frames = data["frames"]
    ff_uv = data["ff_uv"]

    uv1 = ff_uv[:, :2]
    uv2 = ff_uv[:, 2:]
    ff1 = uv_to_3d(uv1, frames)
    ff2 = uv_to_3d(uv2, frames)
    visualize_ff(args.ff, bc, ff1, ff2)
