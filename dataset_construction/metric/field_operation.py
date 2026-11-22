import igl
import os
import numpy as np
import argparse

from metric import vtk_io


def field_visualize(field_path, visual_path, scale=1):
    """
    Visualize the field on plane by Z coordinate lifting
    """
    verts, faces, field = vtk_io.read_vtk_with_attribute(field_path)
    verts[:, 2] = field * scale
    os.makedirs(os.path.dirname(visual_path), exist_ok=True)
    vtk_io.write_vtk_with_attribute(visual_path, verts, faces, field)


def field_visualize_normal(field_path, visual_path, scale=1):
    """
    Visualize the field on plane by normal direction offsetting
    """
    verts, faces, field = vtk_io.read_vtk_with_attribute(field_path)
    normal = igl.per_vertex_normals(verts, faces)
    verts += normal * field[:, np.newaxis] * scale
    os.makedirs(os.path.dirname(visual_path), exist_ok=True)
    vtk_io.write_vtk_with_attribute(visual_path, verts, faces, field)


def normalize_field(field):
    f_min = np.min(field)
    f_max = np.max(field)

    # Normalize the field to range [-1, 1]
    field = 2 * (field - f_min) / (f_max - f_min) - 1

    return field


def field_normalize(field_path, visual_path):
    """
    normalize the field into range [-1, 1]
    """
    verts, faces, field = vtk_io.read_vtk_with_attribute(field_path)

    # Normalize the field to range [-1, 1]
    field = normalize_field(field)

    os.makedirs(os.path.dirname(visual_path), exist_ok=True)
    vtk_io.write_vtk_with_attribute(visual_path, verts, faces, field)


def field_obstacle(field_path, visual_path):
    """
    normalize the field into range [0, 1], and set obstacle region only for inner half
    """
    verts, faces, field = vtk_io.read_vtk_with_attribute(field_path)

    # Normalize the field to range [-1, 1]
    field = normalize_field_as_obstacle(verts, field)

    os.makedirs(os.path.dirname(visual_path), exist_ok=True)
    vtk_io.write_vtk_with_attribute(visual_path, verts, faces, field)


def normalize_field_as_obstacle(V, field):
    # scale from [-1, 1] to [0, 1]
    field = (field + 1) / 2

    # set obstacle only for |x| <=0.25 and |y| <=0.25
    obstacle_indices = np.where(
        (np.abs(V[:, 0]) > 0.25) | (np.abs(V[:, 1]) > 0.25))[0]
    field[obstacle_indices] = 0.0

    return field


def normalize_field_boundary(V, field):
    # scale to [0, 1]
    field = (field - np.min(field)) / (np.max(field) - np.min(field))

    # Define the "safe" inner region and the outer boundary
    inner_limit = 0.25
    outer_limit = 0.50

    # --- Helper: 1D Smoothstep Function ---
    def get_1d_mask(coordinates):
        # 1. Get distance from 0
        d = np.abs(coordinates)

        # 2. Normalize distance to 0..1 range within the transition band
        # If d < 0.25 -> t = 0
        # If d > 0.50 -> t = 1
        t = (d - inner_limit) / (outer_limit - inner_limit)
        t = np.clip(t, 0.0, 1.0)

        # 3. Calculate smooth falloff (1.0 down to 0.0)
        # Using standard cubic smoothstep: 3t^2 - 2t^3
        falloff = 3 * t**2 - 2 * t**3
        return 1.0 - falloff

    # --- Apply Separately ---
    # Calculate mask based on X position
    mask_x = get_1d_mask(V[:, 0])

    # Calculate mask based on Y position
    mask_y = get_1d_mask(V[:, 1])

    # --- Combine by Multiplication ---
    # The product of two smooth functions is always smooth.
    # Inner region: 1.0 * 1.0 = 1.0
    # Boundary:     0.0 * anything = 0.0
    final_mask = mask_x * mask_y

    masked_field = field * final_mask

    current_max = np.max(masked_field)
    current_min = np.min(masked_field)  # Should be 0.0 ideally

    # Avoid dividing by zero if the mask wiped out everything
    if (current_max - current_min) > 1e-12:
        final_field = (masked_field - current_min) / \
            (current_max - current_min)
    else:
        final_field = masked_field  # Field is likely all zeros

    return final_field


def field_obstacle_boundary(field_path, visual_path):
    """
    normalize the field into range [0, 1], and smoothly reduce to zero towards boundary
    """
    verts, faces, field = vtk_io.read_vtk_with_attribute(field_path)

    # Normalize the field to range [0, 1]
    field = normalize_field_boundary(verts, field)

    os.makedirs(os.path.dirname(visual_path), exist_ok=True)
    vtk_io.write_vtk_with_attribute(visual_path, verts, faces, field)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vtk", type=str)
    args.add_argument("--obj", type=str)
    args.add_argument("--scale", type=float, default=1)
    args = args.parse_args()

    field_visualize(args.vtk, args.obj, args.scale)
