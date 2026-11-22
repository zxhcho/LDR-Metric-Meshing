import torch
import os
import igl
import scipy
import numpy as np
import vtk_io

from model.diffusion_net import utils, geometry


def save_op_cache(op_cache_path: str,
                  data: dict):
    os.makedirs(os.path.dirname(op_cache_path), exist_ok=True)

    L_np = utils.sparse_torch_to_np(data["L"])
    gradX_np = utils.sparse_torch_to_np(data["gradX"])
    gradY_np = utils.sparse_torch_to_np(data["gradY"])

    np.savez_compressed(
        op_cache_path,
        verts=data["verts"].detach().cpu().numpy(),
        frames=data["frames"].detach().cpu().numpy(),
        massvec=data["massvec"].detach().cpu().numpy(),
        evals=data["evals"].detach().cpu().numpy(),
        evecs=data["evecs"].detach().cpu().numpy(),

        faces=data["faces"].detach().cpu().numpy().astype(np.int64),

        L_data=L_np.data,
        L_indices=L_np.indices,
        L_indptr=L_np.indptr,
        L_shape=L_np.shape,

        gradX_data=gradX_np.data,
        gradX_indices=gradX_np.indices,
        gradX_indptr=gradX_np.indptr,
        gradX_shape=gradX_np.shape,

        gradY_data=gradY_np.data,
        gradY_indices=gradY_np.indices,
        gradY_indptr=gradY_np.indptr,
        gradY_shape=gradY_np.shape,
    )
    return


def save_feature_cache(feature_cache_path: str,
                       feature: torch.Tensor):
    os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
    np.savez_compressed(feature_cache_path, feature=feature.numpy())


def read_sp_mat(
        prefix: str,
        npzfile: str):
    data = npzfile[prefix + "_data"]
    indices = npzfile[prefix + "_indices"]
    indptr = npzfile[prefix + "_indptr"]
    shape = npzfile[prefix + "_shape"]
    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat


def load_op_cache(
        op_cache_path: str,
        load_feat=True) -> dict:
    npzfile = np.load(op_cache_path, allow_pickle=True)
    data = {}

    ftype = torch.float32

    # dense
    if load_feat:
        data["feature"] = torch.from_numpy(npzfile["feature"]).to(dtype=ftype)
    data["verts"] = torch.from_numpy(npzfile["verts"]).to(dtype=ftype)
    data["frames"] = torch.from_numpy(
        npzfile["frames"]).to(dtype=ftype)
    data["massvec"] = torch.from_numpy(
        npzfile["massvec"]).to(dtype=ftype)
    data["evals"] = torch.from_numpy(npzfile["evals"]).to(dtype=ftype)
    data["evecs"] = torch.from_numpy(npzfile["evecs"]).to(dtype=ftype)

    data["faces"] = torch.from_numpy(npzfile["faces"]).to(dtype=torch.long)

    # sparse
    data["L"] = utils.sparse_np_to_torch(
        read_sp_mat("L", npzfile))
    data["gradX"] = utils.sparse_np_to_torch(
        read_sp_mat("gradX", npzfile))
    data["gradY"] = utils.sparse_np_to_torch(
        read_sp_mat("gradY", npzfile))

    return data


def compute_op_cache(
        filepath: str) -> dict:
    mesh_data = np.load(filepath, allow_pickle=True)
    verts = torch.from_numpy(mesh_data["verts"])
    faces = torch.from_numpy(mesh_data["faces"])

    frames, massvec, L, evals, evecs, gradX, gradY = geometry.compute_operators(
        verts, faces, k_eig=128)
    data = {
        # float
        "verts": verts.to(dtype=torch.float32),
        "frames": frames.to(dtype=torch.float32),
        "massvec": massvec.to(dtype=torch.float32),
        "evals": evals.to(dtype=torch.float32),
        "evecs": evecs.to(dtype=torch.float32),
        # int
        "faces": faces.to(dtype=torch.int64),
        # sparse
        "L": L.to(dtype=torch.float32),
        "gradX": gradX.to(dtype=torch.float32),
        "gradY": gradY.to(dtype=torch.float32),
    }
    return data
