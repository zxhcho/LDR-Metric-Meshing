from copy import deepcopy
import numpy as np

from geometric_kernels.spaces import Mesh
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.kernels import default_feature_map
from geometric_kernels.sampling import sampler

from gstools import SRF, Exponential, Gaussian, Matern

import argparse
import time
import igl

import deepxde as dde

from metric import vtk_io


def GRF_matern(kernel, nv, scale=1.0, nu=3/2, seed=42):
    params = kernel.init_params()
    params["lengthscale"] = np.array([scale])
    params["nu"] = np.array([nu])

    # generate seed by system time
    key = np.random.RandomState(seed)
    all_points = np.arange(nv)[:, None]

    feature_map = default_feature_map(kernel=kernel)
    sample_paths = sampler(feature_map)
    key, samples = sample_paths(all_points, params, key=key)
    return samples


def GRF_grid(vertices, step=100, scale=0.1, batch=1, seed=42):
    """
        Interplolate the Gaussian random field in [0, 1]x[0, 1] on mesh
    """
    np.random.RandomState(seed)
    space = dde.data.GRF2D(length_scale=scale, N=step)
    features = space.random(batch)
    u = space.eval_batch(features, vertices)
    return u


def GRF_aniso(mesh, var, len_scale, angles, seed):

    model = Gaussian(dim=2, var=var, len_scale=len_scale, angles=angles)
    srf = SRF(model, seed=seed)
    x = mesh.vertices[:, 0] * 100
    y = mesh.vertices[:, 1] * 100
    field = srf([x, y], seed=seed)
    return np.array(field).reshape(-1)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mesh", type=str)
    args.add_argument("--scale", type=float, default=0.1)
    args.add_argument("--method", type=str, default="matern")
    args = args.parse_args()

    method = args.method
    scale = args.scale
    mesh_path = args.mesh
    mesh = Mesh.load_mesh(mesh_path)
    seed = int(time.time())
    if method == "matern":
        print("Number of vertices in the mesh:", mesh.num_vertices)

        kernel = MaternGeometricKernel(mesh)
        samples = GRF_matern(kernel, mesh.num_vertices, scale, 3/2, seed)
        print(np.mean(samples))
        print(np.var(samples))

        vtk_io.write_vtk_with_attribute(
            mesh_path + f"_{scale}.vtk", mesh.vertices, mesh.faces, samples.reshape(-1))
    elif method == "grid":
        samples = GRF_grid(mesh, step=100, scale=0.1, batch=1, seed=seed)
        print(np.mean(samples))
        print(np.var(samples))
        vtk_io.write_vtk_with_attribute(
            mesh_path + "_sample.vtk", mesh.vertices, mesh.faces, samples.reshape(-1))
    elif method == "aniso":
        field = GRF_aniso(mesh, seed)
        print(np.mean(field))
        print(np.var(field))
        mesh.vertices[:, 2] = field
        vtk_io.write_vtk_with_attribute(
            mesh_path + "_sample.vtk", mesh.vertices, mesh.faces, field.reshape(-1))
