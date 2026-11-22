#pragma once

#include <Eigen/Dense>
#include <igl/signed_distance.h>
#include <igl/barycentric_coordinates.h>
#include <igl/AABB.h> /* Recommended for static meshes to avoid rebuilding tree */

class ObstacleProjector
{
private:
    Eigen::MatrixXd V_obs;
    Eigen::MatrixXi F_obs;
    Eigen::VectorXd u_obs;

    // We use AABB directly because the obstacle mesh is STATIC.
    // This is much faster than calling signed_distance() repeatedly
    // which might rebuild the tree.
    igl::AABB<Eigen::MatrixXd, 3> tree;

public:
    ObstacleProjector(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXd &u)
        : V_obs(V), F_obs(F), u_obs(u)
    {
        // Initialize the search tree once
        tree.init(V_obs, F_obs);
        std::cout << "Obstacle Projector initialized with " << V_obs.rows() << " vertices." << std::endl;
    }

    // Input: Current Mesh Vertices (V_query)
    // Output: Projected Obstacle Values at these vertices
    Eigen::VectorXd query(const Eigen::MatrixXd &V_query)
    {
        Eigen::VectorXd values(V_query.rows());

// Parallelize for speed
#pragma omp parallel for
        for (int i = 0; i < V_query.rows(); ++i)
        {
            // 1. Find closest point on Dense Obstacle Mesh
            int fid;
            Eigen::RowVector3d closest_p;
            // squared_distance is the core of AABB search
            tree.squared_distance(V_obs, F_obs, V_query.row(i), fid, closest_p);

            // 2. Compute Barycentric Coords on that face
            Eigen::RowVector3d v0 = V_obs.row(F_obs(fid, 0));
            Eigen::RowVector3d v1 = V_obs.row(F_obs(fid, 1));
            Eigen::RowVector3d v2 = V_obs.row(F_obs(fid, 2));
            Eigen::RowVector3d bc;
            igl::barycentric_coordinates(closest_p, v0, v1, v2, bc);

            // 3. Interpolate Obstacle Value
            values(i) = bc(0) * u_obs(F_obs(fid, 0)) +
                        bc(1) * u_obs(F_obs(fid, 1)) +
                        bc(2) * u_obs(F_obs(fid, 2));
        }
        return values;
    }
};