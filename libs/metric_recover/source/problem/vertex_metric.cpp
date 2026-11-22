#include <metric_recover/problem/vertex_metric.h>
#include <metric_recover/function/metric_func.h>

#include <metric_utils/metric_op.h>

#include <Eigen/Eigenvalues>
#include <igl/adjacency_list.h>
#include <igl/barycenter.h>

#include <vector>
#include <list>
#include <numeric>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <lsqcpp/lsqcpp.hpp>

namespace zxh::metric_recover {

int vertex_dof(const Eigen::MatrixXd &edges) {
    //* get the row-rank of the matrix edges with shape (ne, 2), which is not a square matrix
    const int ne = edges.rows();
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    A.resize(ne, 3);
    for (int j = 0; j < ne; ++j) {
        A(j, 0) = edges(j, 0) * edges(j, 0);
        A(j, 1) = 2 * edges(j, 0) * edges(j, 1);
        A(j, 2) = edges(j, 1) * edges(j, 1);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(A.transpose() * A);
    if (eigensolver.info() != 0) {
        spdlog::error("Eigen solver failed!");
        exit(__LINE__);
    }
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    eigenvalues /= eigenvalues.cwiseAbs().maxCoeff();

    int row_rank = 0;
    for (int i = 0; i < 3; ++i) {
        if (fabs(eigenvalues(i)) > 1e-4) {
            row_rank++;
        } else {
            spdlog::debug("Eigenvalues: {},{},{}", eigenvalues(0), eigenvalues(1), eigenvalues(2));
        }
    }

    return row_rank;
}

//* solve linear system to get initial guess
Eigen::Vector3d solve_initial_guess(
    const Eigen::MatrixXd &edges,
    const Eigen::VectorXd &lengths) {
    const int n_ve = lengths.size();
    Eigen::Vector3d xval;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    A.resize(n_ve, 3);
    b.resize(n_ve);
    for (int j = 0; j < n_ve; ++j) {
        A(j, 0) = edges(j, 0) * edges(j, 0);
        A(j, 1) = 2 * edges(j, 0) * edges(j, 1);
        A(j, 2) = edges(j, 1) * edges(j, 1);
        b(j) = lengths(j);
    }
    xval = A.colPivHouseholderQr().solve(b);

    return xval;
}

std::vector<int> k_ring_neighbors(
    const int start,
    const int k,
    const std::vector<std::vector<int>> &VV) {
    std::vector<int> k_ring;
    const int nv = VV.size();

    std::list<std::pair<int, int>> queue;
    std::vector<bool> visited(nv, false);
    queue.push_back(std::pair<int, int>(start, 0));
    visited[start] = true;
    while (!queue.empty()) {
        int toVisit = queue.front().first;
        int distance = queue.front().second;
        queue.pop_front();
        k_ring.push_back(toVisit);
        if (toVisit < nv) {
            if (distance < (int)k) {
                for (unsigned int i = 0; i < VV[toVisit].size(); ++i) {
                    int neighbor = VV[toVisit][i];
                    if (!visited[neighbor]) {
                        queue.push_back(std::pair<int, int>(neighbor, distance + 1));
                        visited[neighbor] = true;
                    }
                }
            }
        }
    }

    return k_ring;
}

int vertex_metric_solver_edge(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric) {
    const int nv = mesh.nv();
    const int ne = mesh.ne();
    assert(edge_lens.rows() == ne);

    const auto &VE = mesh.topology.VE;
    const auto &EV = mesh.topology.EV;

    metric.resize(nv, 4);

    //* guarantee the SPD matrix
    lsqcpp::GaussNewton<double, 3, Eigen::Dynamic, metric_func_spd_t, lsqcpp::ArmijoBacktracking, lsqcpp::DenseCholeskySolver> solver_spd;
    solver_spd.setMaximumIterations(100);
    solver_spd.setMinimumGradientLength(1e-20);
    solver_spd.setMinimumStepLength(1e-20);
    solver_spd.setMinimumError(1e-20);
    solver_spd.setVerbosity(0);

    std::vector<double> init_errors, spd_errors;
    std::vector<int> is_under_determined(nv, 0);
    for (int i = 0; i < nv; ++i) {
        const int n_ve = VE[i].size();
        Eigen::MatrixXd edges;
        Eigen::VectorXd metric_edge_lens;
        edges.resize(n_ve, 2);
        metric_edge_lens.resize(n_ve);
        for (int j = 0; j < n_ve; ++j) {
            const int eid = VE[i][j];
            metric_edge_lens(j) = edge_lens(eid);
            const int v0 = EV(eid, 0);
            const int v1 = EV(eid, 1);
            edges(j, 0) = mesh.V(v1, 0) - mesh.V(v0, 0);
            edges(j, 1) = mesh.V(v1, 1) - mesh.V(v0, 1);
        }

        const int dof = vertex_dof(edges);
        if (dof < 3) {
            is_under_determined[i] = 1;
            spdlog::debug("{} is under determined {}", i, dof);
            continue;
        }

        Eigen::Vector3d xval = solve_initial_guess(edges, metric_edge_lens);

        Eigen::Matrix2d metric_mat;
        metric_mat << xval(0), xval(1),
            xval(1), xval(2);
        xval = metric_mat_to_eig_space(metric_mat);

        solver_spd.setObjective(metric_func_spd_t(edges, metric_edge_lens));
        const auto result_spd = solver_spd.minimize(xval);
        xval = result_spd.xval;
        spd_errors.push_back(result_spd.error);
        metric_mat = metric_eig_space_to_mat(xval);
        metric.row(i) << metric_mat(0, 0), metric_mat(0, 1), metric_mat(1, 0), metric_mat(1, 1);
    }

    double spd_error_norm = std::accumulate(spd_errors.begin(), spd_errors.end(), 0.0) / nv;
    spdlog::debug("Residual SPD: {}", spd_error_norm);

    std::vector<std::vector<int>> VV;
    igl::adjacency_list(mesh.F, VV);

    //* use neighbor metric for under-detemined vertex
    for (int vi = 0; vi < nv; ++vi) {
        if (is_under_determined[vi]) {
            std::vector<int> k_ring = k_ring_neighbors(vi, 3, VV);
            bool under_determined_solved = false;
            for (int v_nei : k_ring) {
                if (!is_under_determined[v_nei]) {
                    metric.row(vi) = metric.row(v_nei);
                    under_determined_solved = true;
                    spdlog::debug("Under determined {} is resolved by {}!", vi, v_nei);
                    break;
                }
            }
            if (!under_determined_solved) {
                spdlog::error("Under determined {} is sill not resolved!", vi);
            }
        }
    }

    return 0;
}

int vertex_metric_solver_dual_edge(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric) {
    const int nv = mesh.nv();
    const int ne = mesh.ne();
    assert(edge_lens.rows() == ne);

    const auto &VE = mesh.topology.VE;
    const auto &EV = mesh.topology.EV;
    const auto &EF = mesh.topology.EF;
    const auto &V = mesh.V;
    const auto &F = mesh.F;
    Eigen::MatrixXd BC;
    igl::barycenter(V, F, BC);

    metric.resize(nv, 4);

    //* guarantee the SPD matrix
    lsqcpp::GaussNewton<double, 3, Eigen::Dynamic, metric_func_spd_t, lsqcpp::ArmijoBacktracking, lsqcpp::DenseCholeskySolver> solver_spd;
    solver_spd.setMaximumIterations(100);
    solver_spd.setMinimumGradientLength(1e-20);
    solver_spd.setMinimumStepLength(1e-20);
    solver_spd.setMinimumError(1e-20);
    solver_spd.setVerbosity(0);

    std::vector<double> init_errors, spd_errors;
    std::vector<int> is_under_determined(nv, 0);
    for (int i = 0; i < nv; ++i) {
        const int n_ve = VE[i].size();
        Eigen::MatrixXd edges;
        Eigen::VectorXd metric_edge_lens;
        edges.resize(n_ve, 2);
        metric_edge_lens.resize(n_ve);
        for (int j = 0; j < n_ve; ++j) {
            const int eid = VE[i][j];
            metric_edge_lens(j) = edge_lens(eid);
            const int f0 = EF(eid, 0);
            const int f1 = EF(eid, 1);
            if (f0 == -1 || f1 == -1) {
                continue;
            }
            edges(j, 0) = BC(f0, 0) - BC(f1, 0);
            edges(j, 1) = BC(f0, 1) - BC(f1, 1);
        }

        const int dof = vertex_dof(edges);
        if (dof < 3) {
            is_under_determined[i] = 1;
            spdlog::debug("{} is under determined {}", i, dof);
            continue;
        }

        Eigen::Vector3d xval = solve_initial_guess(edges, metric_edge_lens);

        Eigen::Matrix2d metric_mat;
        metric_mat << xval(0), xval(1),
            xval(1), xval(2);
        xval = metric_mat_to_eig_space(metric_mat);

        solver_spd.setObjective(metric_func_spd_t(edges, metric_edge_lens));
        const auto result_spd = solver_spd.minimize(xval);
        xval = result_spd.xval;
        spd_errors.push_back(result_spd.error);
        metric_mat = metric_eig_space_to_mat(xval);
        metric.row(i) << metric_mat(0, 0), metric_mat(0, 1), metric_mat(1, 0), metric_mat(1, 1);
    }

    double spd_error_norm = std::accumulate(spd_errors.begin(), spd_errors.end(), 0.0) / nv;
    spdlog::debug("Residual SPD: {}", spd_error_norm);

    std::vector<std::vector<int>> VV;
    igl::adjacency_list(mesh.F, VV);

    //* use neighbor metric for under-detemined vertex
    for (int vi = 0; vi < nv; ++vi) {
        if (is_under_determined[vi]) {
            std::vector<int> k_ring = k_ring_neighbors(vi, 3, VV);
            bool under_determined_solved = false;
            for (int v_nei : k_ring) {
                if (!is_under_determined[v_nei]) {
                    metric.row(vi) = metric.row(v_nei);
                    under_determined_solved = true;
                    spdlog::debug("Under determined {} is resolved by {}!", vi, v_nei);
                    break;
                }
            }
            if (!under_determined_solved) {
                spdlog::error("Under determined {} is sill not resolved!", vi);
            }
        }
    }

    return 0;
}

int get_tanget_basis(
    const Eigen::MatrixXd &N,
    Eigen::MatrixXd &B1,
    Eigen::MatrixXd &B2) {
    const int n = N.rows();
    B1.resize(n, 3);
    B2.resize(n, 3);

    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d normal = N.row(i).normalized();
        Eigen::Vector3d b1 = normal.cross(Eigen::Vector3d::Random().normalized()).normalized();
        B1.row(i) = b1;
        B2.row(i) = normal.cross(b1).normalized();
    }

    return 0;
}

int vertex_metric_solver_edge_tangent(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric) {
    const int nv = mesh.nv();
    const int ne = mesh.ne();
    assert(edge_lens.rows() == ne);

    const auto &VN = mesh.geometry.VN;
    const auto &VE = mesh.topology.VE;
    const auto &EV = mesh.topology.EV;
    Eigen::MatrixXd metric_tangent;
    metric_tangent.resize(nv, 4);

    Eigen::MatrixXd B1, B2;
    get_tanget_basis(VN, B1, B2);
    assert(B1.rows() == nv);
    assert(B2.rows() == nv);
    assert(B1.cols() == 3);
    assert(B2.cols() == 3);

    //* guarantee the SPD matrix
    lsqcpp::GaussNewton<double, 3, Eigen::Dynamic, metric_func_spd_t, lsqcpp::ArmijoBacktracking, lsqcpp::DenseCholeskySolver> solver_spd;
    solver_spd.setMaximumIterations(100);
    solver_spd.setMinimumGradientLength(1e-20);
    solver_spd.setMinimumStepLength(1e-20);
    solver_spd.setMinimumError(1e-20);
    solver_spd.setVerbosity(0);

    std::vector<double> init_errors, spd_errors;
    const double scale = 1;
    std::vector<int> is_under_determined(nv, 0);
    for (int vi = 0; vi < nv; ++vi) {
        const int n_ve = VE[vi].size();

        Eigen::MatrixXd edges;
        edges.resize(n_ve, 2);

        Eigen::VectorXd metric_edge_lens;
        metric_edge_lens.resize(n_ve);

        const Eigen::Vector3d b1 = B1.row(vi),
                              b2 = B2.row(vi);
        for (int vei = 0; vei < n_ve; ++vei) {
            const int ei = VE[vi][vei];
            metric_edge_lens(vei) = edge_lens(ei) * scale;
            const int v0 = EV(ei, 0);
            const int v1 = EV(ei, 1);
            Eigen::Vector3d edge_vec = mesh.V.row(v1) - mesh.V.row(v0);
            edges(vei, 0) = edge_vec.dot(b1);
            edges(vei, 1) = edge_vec.dot(b2);
        }

        const int dof = vertex_dof(edges);
        if (dof < 3) {
            is_under_determined[vi] = 1;
            spdlog::debug("{} is under determined {}", vi, dof);
            continue;
        }

        Eigen::Vector3d xval = solve_initial_guess(edges, metric_edge_lens);

        Eigen::Matrix2d metric_mat;
        metric_mat << xval(0), xval(1),
            xval(1), xval(2);
        xval = metric_mat_to_eig_space(metric_mat);

        solver_spd.setObjective(metric_func_spd_t(edges, metric_edge_lens));
        const auto result_spd = solver_spd.minimize(xval);
        xval = result_spd.xval;
        spd_errors.push_back(result_spd.error);
        metric_mat = metric_eig_space_to_mat(xval) / scale;
        metric_tangent.row(vi) << metric_mat(0, 0), metric_mat(0, 1), metric_mat(1, 0), metric_mat(1, 1);
    }

    double spd_error_norm = std::accumulate(spd_errors.begin(), spd_errors.end(), 0.0) / nv;
    spdlog::debug("Residual SPD: {}", spd_error_norm);

    //* use neighbor metric for under-detemined vertex
    std::vector<std::vector<int>> VV;
    igl::adjacency_list(mesh.F, VV);

    for (int vi = 0; vi < nv; ++vi) {
        if (is_under_determined[vi]) {
            std::vector<int> k_ring = k_ring_neighbors(vi, 3, VV);
            bool under_determined_solved = false;
            for (int v_nei : k_ring) {
                if (!is_under_determined[v_nei]) {
                    metric_tangent.row(vi) = metric_tangent.row(v_nei);
                    under_determined_solved = true;
                    spdlog::debug("Under determined {} is resolved by {}!", vi, v_nei);
                    break;
                }
            }
            if (!under_determined_solved) {
                spdlog::error("Under determined {} is sill not resolved!", vi);
            }
        }
    }

    metric = zxh::metric_utils::metric_from_tangent_to_ambient(B1, B2, metric_tangent);

    return 0;
}

int vertex_metric_solver_halfedge(
    const mesh_t &mesh,
    const Eigen::MatrixXd &edge_lens,
    Eigen::MatrixXd &metric) {
    const int nv = mesh.nv();
    const int ne = mesh.ne();
    assert(edge_lens.rows() == ne);
    assert(edge_lens.cols() == 2);

    metric.resize(nv, 4);

    //* guarantee the SPD matrix
    lsqcpp::GaussNewton<double, 3, Eigen::Dynamic, metric_func_spd_t, lsqcpp::ArmijoBacktracking, lsqcpp::DenseCholeskySolver> solver_spd;
    solver_spd.setMaximumIterations(100);
    solver_spd.setMinimumGradientLength(1e-20);
    solver_spd.setMinimumStepLength(1e-20);
    solver_spd.setMinimumError(1e-20);
    solver_spd.setVerbosity(0);

    const auto &VE = mesh.topology.VE;
    const auto &EV = mesh.topology.EV;

    std::vector<double> init_errors, spd_errors;
    for (int i = 0; i < nv; ++i) {
        const int n_ve = VE[i].size();
        Eigen::MatrixXd edges;
        Eigen::VectorXd metric_edge_lens;
        edges.resize(n_ve, 2);
        metric_edge_lens.resize(n_ve);
        for (int j = 0; j < n_ve; ++j) {
            const int eid = VE[i][j];
            const int veidx = (i == EV(eid, 0)) ? 0 : 1;
            metric_edge_lens(j) = edge_lens(eid, veidx);
            const int v0 = EV(eid, 0);
            const int v1 = EV(eid, 1);
            edges(j, 0) = mesh.V(v1, 0) - mesh.V(v0, 0);
            edges(j, 1) = mesh.V(v1, 1) - mesh.V(v0, 1);
        }

        Eigen::Vector3d xval = solve_initial_guess(edges, metric_edge_lens);

        //* project the initial guess to the feasible SPD region
        Eigen::Matrix2d metric_mat;
        metric_mat << xval(0), xval(1),
            xval(1), xval(2);
        xval = metric_mat_to_eig_space(metric_mat);

        //* solve the nonlinear least square problem
        solver_spd.setObjective(metric_func_spd_t(edges, metric_edge_lens));
        const auto result_spd = solver_spd.minimize(xval);
        xval = result_spd.xval;
        spd_errors.push_back(result_spd.error);
        metric_mat = metric_eig_space_to_mat(xval);
        metric.row(i) << metric_mat(0, 0), metric_mat(0, 1), metric_mat(1, 0), metric_mat(1, 1);
    }

    double spd_error_norm = std::accumulate(spd_errors.begin(), spd_errors.end(), 0.0) / nv;
    spdlog::debug("Residual SPD: {}", spd_error_norm);

    return 0;
}

int vertex_metric_solver_dual_edge_tangent(
    const mesh_t &mesh,
    const Eigen::VectorXd &edge_lens,
    Eigen::MatrixXd &metric) {
    return 0;
}
} // namespace zxh::metric_recover