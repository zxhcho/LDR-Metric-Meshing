#pragma once

#include <igl/barycenter.h>
#include <igl/barycentric_coordinates.h>
#include <igl/doublearea.h>
#include <igl/signed_distance.h>

#include <Eigen/Dense>

class ErrorEvaluator
{
private:
  // --- Static Ground Truth Data ---
  Eigen::MatrixXd V_gt;
  Eigen::MatrixXi F_gt;
  Eigen::VectorXd u_gt, areas_gt;
  double l1_norm_gt = 0.0;
  Eigen::MatrixXd P_gt; // GT Barycenters (query points)

  // --- Dynamic Calculation Mesh Data ---
  // We must store a copy of the current faces F_calc to know
  // which nodes to interpolate from during compute_rel_l1
  Eigen::MatrixXi F_calc_current;

  // --- Dynamic Mapping Data ---
  Eigen::VectorXi
      mapped_face_indices;            // Which calc-face contains the GT point?
  Eigen::MatrixXd mapped_bary_coords; // Where in that face is it?

public:
  // 1. Constructor: Initialize Static GT Data
  ErrorEvaluator(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                 const Eigen::VectorXd &u)
      : V_gt(V), F_gt(F), u_gt(u)
  {
    // Compute GT Areas
    igl::doublearea(V_gt, F_gt, areas_gt);
    areas_gt *= 0.5;

    // Compute GT Norm
    for (int i = 0; i < F_gt.rows(); ++i)
    {
      double val =
          (u_gt(F_gt(i, 0)) + u_gt(F_gt(i, 1)) + u_gt(F_gt(i, 2))) / 3.0;
      l1_norm_gt += std::abs(val) * areas_gt(i);
    }

    // Precompute GT barycenters (These are our query points, they never change)
    igl::barycenter(V_gt, F_gt, P_gt);
  }

  // 2. Update API: Call this whenever the mesh (V_c, F_c) changes!
  void update_geometry(const Eigen::MatrixXd &V_c, const Eigen::MatrixXi &F_c)
  {
    // A. Store the new topology
    F_calc_current = F_c;

    // B. Re-run Spatial Search (Expensive, but necessary on adaptation)
    Eigen::VectorXd S;
    Eigen::MatrixXd C, N;
    // This rebuilds the AABB tree for the new mesh V_c, F_c
    igl::signed_distance(P_gt, V_c, F_c, igl::SIGNED_DISTANCE_TYPE_DEFAULT, S,
                         mapped_face_indices, C, N);

    // C. Update Barycentric Coordinates
    mapped_bary_coords.resize(P_gt.rows(), 3);

    Eigen::RowVector3d v0, v1, v2, p, bc;

    for (int i = 0; i < P_gt.rows(); ++i)
    {
      int fid = mapped_face_indices(i);

      // Retrieve vertices of the CLOSEST face on the NEW mesh
      v0 = V_c.row(F_c(fid, 0));
      v1 = V_c.row(F_c(fid, 1));
      v2 = V_c.row(F_c(fid, 2));
      p = C.row(i);

      igl::barycentric_coordinates(p, v0, v1, v2, bc);
      mapped_bary_coords.row(i) = bc;
    }
  }

  // 3. Evaluate: Call this every iteration (Fast)
  double compute_rel_l1(const Eigen::VectorXd &u_calc)
  {
    // Sanity Check
    if (u_calc.size() == 0 || mapped_face_indices.size() == 0)
      return 0.0;

    double l1_diff = 0.0;
    int n_gt = F_gt.rows();

    for (int i = 0; i < n_gt; ++i)
    {
      int fid = mapped_face_indices(i);

      // Interpolate using the STORED F_calc_current
      // (Indices here are valid for the u_calc vector passed in)
      double val_calc =
          mapped_bary_coords(i, 0) * u_calc(F_calc_current(fid, 0)) +
          mapped_bary_coords(i, 1) * u_calc(F_calc_current(fid, 1)) +
          mapped_bary_coords(i, 2) * u_calc(F_calc_current(fid, 2));

      double val_gt =
          (u_gt(F_gt(i, 0)) + u_gt(F_gt(i, 1)) + u_gt(F_gt(i, 2))) / 3.0;

      l1_diff += std::abs(val_calc - val_gt) * areas_gt(i);
    }

    return l1_diff / l1_norm_gt;
  }
};