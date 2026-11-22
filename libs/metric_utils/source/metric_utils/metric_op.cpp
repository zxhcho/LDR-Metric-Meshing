#include <metric_utils/metric_op.h>

#include <spdlog/spdlog.h>

#include <Eigen/Eigenvalues>

#include <igl/principal_curvature.h>
#include <igl/adjacency_list.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/barycenter.h>

#include <vector>
#include <set>

namespace zxh::metric_utils {

Eigen::MatrixXd metric_from_tangent_to_ambient(
    const Eigen::MatrixXd &B1,
    const Eigen::MatrixXd &B2,
    const Eigen::MatrixXd &metric) {
    const int nm = metric.rows();
    assert(B1.rows() == nm);
    assert(B2.rows() == nm);
    assert(metric.cols() == 4);

    Eigen::MatrixXd metric_ambient;
    metric_ambient.resize(nm, 9);
    for (int i = 0; i < nm; i++) {
        Eigen::Matrix2d m;
        m << metric(i, 0), metric(i, 1),
            metric(i, 2), metric(i, 3);
        const Eigen::Vector3d b1 = B1.row(i),
                              b2 = B2.row(i);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(m);
        if (eigensolver.info() != Eigen::Success) {
            spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
            exit(1);
        }
        const Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
        const Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();

        const double eps = 1e-20;
        const Eigen::Vector3d eig_vals_ambient = Eigen::Vector3d(
            (fabs(eigenvalues(0))), (fabs(eigenvalues(1))), eps);
        Eigen::Matrix3d eig_vecs_ambient = Eigen::Matrix3d::Zero();
        eig_vecs_ambient.col(0) = eigenvectors(0, 0) * b1 + eigenvectors(1, 0) * b2;
        eig_vecs_ambient.col(1) = eigenvectors(0, 1) * b1 + eigenvectors(1, 1) * b2;
        eig_vecs_ambient.col(2) = b1.cross(b2).normalized();

        Eigen::Matrix3d metric_ambient_i =
            eig_vecs_ambient * eig_vals_ambient.asDiagonal() *
            eig_vecs_ambient.transpose();
        metric_ambient.row(i) = Eigen::Map<Eigen::RowVectorXd>(
            metric_ambient_i.data(), metric_ambient_i.size());
    }

    return metric_ambient;
}

Eigen::MatrixXd metric_face_to_vertex(
    const std::vector<std::vector<int>> &VF,
    const Eigen::MatrixXd &metric) {
    const int nv = VF.size();
    assert(metric.cols() == 3 * 3);

    Eigen::MatrixXd metric_vertex;
    metric_vertex.resize(nv, 3 * 3);
    for (int i = 0; i < nv; i++) {
        const auto &faces = VF[i];
        const int nf = faces.size();
        Eigen::VectorXd metric_vertex_i = Eigen::VectorXd::Zero(9);
        for (int j = 0; j < nf; j++) {
            const int f = faces[j];
            metric_vertex_i += metric.row(f) / nf;
        }
        metric_vertex.row(i) = metric_vertex_i;
    }

    return metric_vertex;
}

Eigen::MatrixXd principal_curvature_as_metric(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const int k = 5) {
    const int nf = F.rows();
    const int nv = V.rows();

    Eigen::MatrixXd metric;
    metric.resize(nv, 9);

    Eigen::MatrixXd PD1, PD2;
    Eigen::VectorXd PV1, PV2;
    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2, k);

    for (int i = 0; i < nv; i++) {
        const Eigen::Vector3d eig_vals_ambient = Eigen::Vector3d(
            (fabs(PV1(i))), (fabs(PV2(i))), 1.0e-10);
        Eigen::Matrix3d eig_vecs_ambient = Eigen::Matrix3d::Zero();
        eig_vecs_ambient.col(0) = PD1.row(i).normalized();
        eig_vecs_ambient.col(1) = PD2.row(i).normalized();
        eig_vecs_ambient.col(2) =
            eig_vecs_ambient.col(0).cross(eig_vecs_ambient.col(1)).normalized();

        Eigen::Matrix3d metric_ambient_i =
            eig_vecs_ambient * eig_vals_ambient.asDiagonal() *
            eig_vecs_ambient.transpose();
        metric.row(i) = Eigen::Map<Eigen::RowVectorXd>(
            metric_ambient_i.data(), metric_ambient_i.size());
    }

    return metric;
}

//* Find the k-ring vertex neighbors for all vertices
std::vector<std::vector<int>> k_ring_vertices(
    const Eigen::MatrixXi &F,
    const int k) {
    std::vector<std::vector<int>> VV;
    igl::adjacency_list(F, VV);
    const int nv = VV.size();
    std::vector<std::vector<int>> k_ring(nv);

    std::vector<bool> visited(nv, false);
    for (int i = 0; i < nv; i++) {
        const int start = i;
        visited.assign(nv, false);
        visited[start] = true;

        std::list<std::pair<int, int>> queue;
        queue.push_back(std::pair<int, int>(start, 0));
        while (!queue.empty()) {
            int toVisit = queue.front().first;
            int distance = queue.front().second;
            queue.pop_front();
            k_ring[i].push_back(toVisit);
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
    }

    return k_ring;
}

//* Find the k-ring face neighbors for all vertices
std::vector<std::vector<int>> k_ring_faces(
    const Eigen::MatrixXi &F,
    const int nv,
    const int k) {
    int nf = F.rows();

    // 1. Compute Vertex-Face adjacency (VF)
    // To find the initial 1-ring (seed faces) for each vertex
    std::vector<std::vector<int>> VF, VFi;
    igl::vertex_triangle_adjacency(nv, F, VF, VFi);

    // 2. Compute Face-Face adjacency (TT)
    // TT is an #F by 3 matrix. TT(i, j) is the face index adjacent to face i across edge j.
    // -1 indicates a boundary edge.
    Eigen::MatrixXi TT;
    igl::triangle_triangle_adjacency(F, TT);

    std::vector<std::vector<int>> k_ring_F(nv);

    // 3. For each vertex, perform BFS on the Dual Graph (faces)
    for (int i = 0; i < nv; i++) {
        // Initial set: The 1-ring faces (incident to the vertex)
        std::vector<int> current_ring = VF[i];

        // Use a set or boolean array to track visited faces to avoid duplicates/cycles
        // A local set is cleaner for readability, though a persistent bool vector is faster for huge meshes.
        std::set<int> visited(current_ring.begin(), current_ring.end());

        std::vector<int> frontier = current_ring;

        // Expand k-1 times
        // If k=1, this loop doesn't run, and we return the incident faces.
        for (int step = 0; step < k - 1; ++step) {
            std::vector<int> next_frontier;
            for (int f : frontier) {
                // Check all 3 neighbors of face f
                for (int j = 0; j < 3; ++j) {
                    int neighbor_f = TT(f, j);
                    // -1 means boundary (no neighbor)
                    if (neighbor_f != -1) {
                        if (visited.find(neighbor_f) == visited.end()) {
                            visited.insert(neighbor_f);
                            next_frontier.push_back(neighbor_f);
                            current_ring.push_back(neighbor_f);
                        }
                    }
                }
            }
            if (next_frontier.empty())
                break;
            frontier = next_frontier;
        }

        // // Output is usually expected to be sorted
        // std::sort(current_ring.begin(), current_ring.end());
        k_ring_F[i] = current_ring;
    }

    return k_ring_F;
}

/**
 * @brief Computes a Riemannian metric field from the Hessian of a scalar field defined on a 2D mesh.
 *
 * This function approximates the Hessian of a scalar field $f(x, y)$ at each vertex of a mesh
 * by fitting a local quadratic polynomial to the $k$-ring neighborhood of the vertex.
 *
 * The local quadratic approximation is given by:
 * $f(x, y) \approx ax^2 + bxy + cy^2 + dx + ey + g$
 *
 * The coefficients $(a, b, c, d, e, g)$ are determined by solving a least-squares problem
 * $Ax = b$ using the neighbors' positions and field values.
 *
 * The resulting metric tensor at vertex $v_i$ is the Hessian matrix of the fitted quadratic:
 * $M_i = \begin{pmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{pmatrix} = \begin{pmatrix} 2a & b \\ b & 2c \end{pmatrix}$
 *
 * @param V The vertex positions of the mesh, an $N \times 2$ (or $N \times 3$ with z=0) matrix.
 * @param F The face indices of the mesh, an $M \times 3$ matrix.
 * @param field The scalar field values defined at each vertex, a vector of size $N$.
 * @param k The size of the ring neighborhood to consider for the least-squares fitting (e.g., 1-ring, 2-ring).
 *
 * @return Eigen::MatrixXd An $N \times 4$ matrix where each row represents the flattened $2 \times 2$ metric tensor
 *         $(M_{00}, M_{01}, M_{10}, M_{11})$ for the corresponding vertex.
 *
 * @note The function requires at least 6 neighbors in the $k$-ring to solve for the 6 degrees of freedom
 *       of the quadratic polynomial. If a vertex has fewer neighbors, the program will log a warning and exit.
 */
Eigen::MatrixXd metric_from_hessian_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &field,
    const int k,
    const MetricType m_type) {
    const int nv = V.rows();
    assert(field.size() == nv);

    Eigen::MatrixXd metric;
    metric.resize(nv, 4);

    //* fit a quadratic function according to the k-ring neighbors
    //* and give the hessian of it
    std::vector<std::vector<int>> k_ring = k_ring_vertices(F, k);
    constexpr int dof = 6;
    for (int vi = 0; vi < nv; vi++) {
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        const auto &k_ring_i = k_ring[vi];
        const int k_ring_size = k_ring_i.size();
        A.resize(k_ring_size, dof);
        b.resize(k_ring_size);
        if (k_ring_size < dof) {
            spdlog::warn("Vertex {} has less than 6 neighbors, DOF is not enough.", vi);
            exit(1);
        }
        const double cx = V(vi, 0);
        const double cy = V(vi, 1);
        const double c_val = field(vi);
        for (int j = 0; j < k_ring_size; j++) {
            const int vj = k_ring_i[j];
            const double dx = V(vj, 0) - cx;
            const double dy = V(vj, 1) - cy;
            A.row(j) << dx * dx, dx * dy, dy * dy, dx, dy, 1;
            b(j) = field(vj) - c_val;
        }
        Eigen::VectorXd x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        if (m_type == MetricType::METRIC_P1) {
            double m11 = 2 * x(0);
            double m12 = x(1);
            double m22 = 2 * x(2);
            metric.row(vi) << m11, m12, m12, m22;
        } else if (m_type == MetricType::METRIC_P0) {
            double gx = x(3);
            double gy = x(4);
            double m11 = gx * gx;
            double m12 = gx * gy;
            double m22 = gy * gy;
            metric.row(vi) << m11, m12, m12, m22;
        }
    }
    return metric;
}

Eigen::MatrixXd metric_from_hessian_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXcd &field,
    const int k,
    const MetricType m_type) {
    const int nv = V.rows();

    //* Cross field defined on vertices or faces
    const bool is_vertex_field = (field.size() == nv);

    Eigen::MatrixXd metric_real, metric_img;
    metric_real.resize(nv, 4);
    metric_img.resize(nv, 4);

    //* fit a quadratic function according to the k-ring neighbors
    //* and give the hessian of it
    std::vector<std::vector<int>> k_ring;
    Eigen::MatrixXd BC;
    if (is_vertex_field) {
        k_ring = k_ring_vertices(F, k);
    } else {
        k_ring = k_ring_faces(F, nv, k);
        igl::barycenter(V, F, BC);
    }
    constexpr int dof = 6;
    for (int vi = 0; vi < nv; vi++) {
        Eigen::MatrixXd A;
        Eigen::VectorXd b_real, b_img;
        const auto &k_ring_i = k_ring[vi];
        const int k_ring_size = k_ring_i.size();
        A.resize(k_ring_size, dof);
        b_real.resize(k_ring_size);
        b_img.resize(k_ring_size);
        if (k_ring_size < dof) {
            spdlog::warn("Vertex {} has less than 6 neighbors, DOF is not enough.", vi);
            exit(1);
        }
        const double cx = V(vi, 0);
        const double cy = V(vi, 1);

        if (is_vertex_field) {
            const double c_val_real = field(vi).real();
            const double c_val_img = field(vi).imag();
            for (int j = 0; j < k_ring_size; j++) {
                const int vj = k_ring_i[j];
                const double dx = V(vj, 0) - cx;
                const double dy = V(vj, 1) - cy;
                A.row(j) << dx * dx, dx * dy, dy * dy, dx, dy, 1;
                b_real(j) = field(vj).real() - c_val_real;
                b_img(j) = field(vj).imag() - c_val_img;
            }
        } else {
            for (int j = 0; j < k_ring_size; j++) {
                const int fj = k_ring_i[j];
                const double dx = BC(fj, 0) - cx;
                const double dy = BC(fj, 1) - cy;
                A.row(j) << dx * dx, dx * dy, dy * dy, dx, dy, 1;
                b_real(j) = field(fj).real();
                b_img(j) = field(fj).imag();
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd x_real = svd.solve(b_real);
        Eigen::VectorXd x_img = svd.solve(b_img);

        if (m_type == MetricType::METRIC_P1) {
            metric_real.row(vi) << 2 * x_real(0), x_real(1), x_real(1), 2 * x_real(2);
            metric_img.row(vi) << 2 * x_img(0), x_img(1), x_img(1), 2 * x_img(2);
        } else if (m_type == MetricType::METRIC_P0) {
            double gx_real = x_real(3);
            double gy_real = x_real(4);
            double gx_img = x_img(3);
            double gy_img = x_img(4);
            metric_real.row(vi) << gx_real * gx_real, gx_real * gy_real,
                gx_real * gy_real, gy_real * gy_real;
            metric_img.row(vi) << gx_img * gx_img, gx_img * gy_img,
                gx_img * gy_img, gy_img * gy_img;
        }
    }

    Eigen::MatrixXd metric_intersect = (metric_real + metric_img) / 2;
    return metric_intersect;
}

Eigen::MatrixXd metric_from_hessian_3d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &field,
    const int k) {
    const int nv = V.rows();
    assert(field.size() == nv);

    Eigen::MatrixXd metric;
    metric.resize(nv, 9);

    //* fit a quadratic function according to the k-ring neighbors
    //* and give the hessian of it
    std::vector<std::vector<int>> k_ring = k_ring_vertices(F, k);
    constexpr int dof = 10;
    for (int vi = 0; vi < nv; vi++) {
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        const auto &k_ring_i = k_ring[vi];
        const int k_ring_size = k_ring_i.size();
        A.resize(k_ring_size, dof);
        b.resize(k_ring_size);
        if (k_ring_size < dof) {
            spdlog::warn("Vertex {} has less than 6 neighbors, DOF is not enough.", vi);
            exit(1);
        }
        for (int j = 0; j < k_ring_size; j++) {
            const int vj = k_ring_i[j];
            const double x = V(vj, 0);
            const double y = V(vj, 1);
            const double z = V(vj, 2);
            A.row(j) << x * x, x * y, x * z, y * y, y * z, z * z, x, y, z, 1;
            b(j) = field(vj);
        }
        Eigen::VectorXd x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        metric.row(vi) << 2 * x(0), x(1), x(2), x(1), 2 * x(3), x(4), x(2), x(4), 2 * x(5);
    }
    return metric;
}

int metric_spd_2d(Eigen::MatrixXd &metric) {
    const int n = metric.rows();
    const double eps = 1e-8;
    for (int i = 0; i < n; i++) {
        Eigen::Matrix2d m;
        m << metric(i, 0), metric(i, 1),
            metric(i, 2), metric(i, 3);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(m);
        if (eigensolver.info() != Eigen::Success) {
            spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
            return 1;
        }
        Eigen::Vector2d eigenvalues = eigensolver.eigenvalues().cwiseAbs();
        Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();
        m = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose() +
            eps * Eigen::Matrix2d::Identity();
        metric.row(i) = Eigen::Map<Eigen::RowVectorXd>(m.data(), m.size());
    }
    return 0;
}

int metric_spd_3d(Eigen::MatrixXd &metric) {
    const int n = metric.rows();
    const double eps = 1e-8;
    for (int i = 0; i < n; i++) {
        Eigen::Matrix3d m;
        m << metric(i, 0), metric(i, 1), metric(i, 2),
            metric(i, 3), metric(i, 4), metric(i, 5),
            metric(i, 6), metric(i, 7), metric(i, 8);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(m);
        if (eigensolver.info() != Eigen::Success) {
            spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
            return 1;
        }
        Eigen::Vector3d eigenvalues = eigensolver.eigenvalues().cwiseAbs();
        Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();
        m = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose() +
            eps * Eigen::Matrix3d::Identity();
        metric.row(i) = Eigen::Map<Eigen::RowVectorXd>(m.data(), m.size());
    }
    return 0;
}

int metric_spd(Eigen::MatrixXd &metric) {
    assert(metric.cols() == 9 || metric.cols() == 4);
    const int dim = (metric.cols() == 4) ? 2 : 3;
    if (dim == 2) {
        return metric_spd_2d(metric);
    } else {
        return metric_spd_3d(metric);
    }

    return 1;
}

Eigen::MatrixXd metric_2d_to_3d(const Eigen::MatrixXd &metric_2d) {
    assert(metric_2d.cols() == 4);
    Eigen::MatrixXd metric_3d;
    const int n = metric_2d.rows();
    metric_3d.resize(n, 9);
    for (int i = 0; i < n; i++) {
        metric_3d.row(i) << metric_2d(i, 0), metric_2d(i, 1), 0,
            metric_2d(i, 2), metric_2d(i, 3), 0,
            0, 0, 0;
    }
    return metric_3d;
}

Eigen::MatrixXd metric_3d_to_2d(const Eigen::MatrixXd &metric_3d) {
    assert(metric_3d.cols() == 9);
    Eigen::MatrixXd metric_2d;
    const int n = metric_3d.rows();
    metric_2d.resize(n, 4);
    for (int i = 0; i < n; i++) {
        metric_2d.row(i) << metric_3d(i, 0), metric_3d(i, 1),
            metric_3d(i, 3), metric_3d(i, 4);
    }

    return metric_2d;
}

// Helper to force a matrix to be SPD (Symmetric Positive Definite)
// This handles the "near 0" or negative eigenvalue cases from raw Hessians.
//* TODO: set epsilon from outside input
Eigen::Matrix2d make_spd(
    const Eigen::Matrix2d &m,
    const double epsilon) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(m);
    Eigen::Vector2d D = es.eigenvalues();
    Eigen::Matrix2d V = es.eigenvectors();

    // Floor eigenvalues to a small epsilon to prevent singularity
    // This solves the "near 0" explosion in inversion.
    D(0) = (D(0) < epsilon) ? epsilon : D(0);
    D(1) = (D(1) < epsilon) ? epsilon : D(1);

    return V * D.asDiagonal() * V.transpose();
}

int make_spd_batch(
    Eigen::MatrixXd &m,
    const double factor) {

    const int n = m.rows();
    double eig_max = 0;
    for (int i = 0; i < n; i++) {
        Eigen::Matrix2d mi;
        mi << m(i, 0), m(i, 1),
            m(i, 2), m(i, 3);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(mi);
        Eigen::Vector2d D = es.eigenvalues();
        eig_max = std::max(eig_max, D(1));
    }

    for (int i = 0; i < n; i++) {
        Eigen::Matrix2d mi;
        mi << m(i, 0), m(i, 1),
            m(i, 2), m(i, 3);
        Eigen::Matrix2d mi_spd = make_spd(mi, factor * eig_max);
        m.row(i) = Eigen::Map<Eigen::RowVectorXd>(mi_spd.data(), mi_spd.size());
    }

    return 0;
}

Eigen::Matrix2d metric_intersection_2d(
    const Eigen::Matrix2d &a,
    const Eigen::Matrix2d &b) {

    const Eigen::Matrix2d N = a.inverse() * b;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(N);
    if (eigensolver.info() != Eigen::Success) {
        spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
        spdlog::error("Matrix a: \n {}, {}, {}", a(0, 0), a(0, 1), a(1, 1));
        spdlog::error("Matrix b: \n {}, {}, {}", b(0, 0), b(0, 1), b(1, 1));
        exit(1);
    }
    const auto P = eigensolver.eigenvectors();

    const Eigen::Matrix2d D1 = P.transpose() * a * P;
    const Eigen::Matrix2d D2 = P.transpose() * b * P;
    const auto diag_d1 = D1.diagonal();
    const auto diag_d2 = D2.diagonal();
    const auto diag_D = (diag_d1.array() > diag_d2.array()).select(diag_d1, diag_d2).asDiagonal();

    Eigen::Matrix2d m = P.inverse().transpose() * diag_D * P.inverse();
    return m;
}

Eigen::Matrix3d metric_intersection_3d(
    const Eigen::Matrix3d &a,
    const Eigen::Matrix3d &b) {
    const Eigen::Matrix3d N = a.inverse() * b;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(N);
    if (eigensolver.info() != Eigen::Success) {
        spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
        exit(1);
    }
    const auto P = eigensolver.eigenvectors();

    const Eigen::Matrix3d D1 = P.transpose() * a * P;
    const Eigen::Matrix3d D2 = P.transpose() * b * P;
    const auto diag_d1 = D1.diagonal();
    const auto diag_d2 = D2.diagonal();
    const auto diag_D = (diag_d1.array() > diag_d2.array()).select(diag_d1, diag_d2).asDiagonal();

    Eigen::Matrix3d m = P.inverse().transpose() * diag_D * P.inverse();
    return m;
}

Eigen::MatrixXd metric_intersection_batch(
    const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    assert(a.cols() == b.cols());
    Eigen::MatrixXd m;
    const int n = a.rows();
    const int dim = a.cols();
    m.resize(n, dim);
    for (int i = 0; i < n; i++) {
        if (dim == 4) {
            Eigen::Matrix2d a_i, b_i;
            a_i << a(i, 0), a(i, 1), a(i, 2), a(i, 3);
            b_i << b(i, 0), b(i, 1), b(i, 2), b(i, 3);
            Eigen::Matrix2d m_i = metric_intersection_2d(a_i, b_i);
            m.row(i) = Eigen::Map<Eigen::RowVectorXd>(m_i.data(), m_i.size());
        } else if (dim == 9) {
            Eigen::Matrix3d a_i, b_i;
            a_i << a(i, 0), a(i, 1), a(i, 2),
                a(i, 3), a(i, 4), a(i, 5),
                a(i, 6), a(i, 7), a(i, 8);
            b_i << b(i, 0), b(i, 1), b(i, 2),
                b(i, 3), b(i, 4), b(i, 5),
                b(i, 6), b(i, 7), b(i, 8);
            Eigen::Matrix3d m_i = metric_intersection_3d(a_i, b_i);
            m.row(i) = Eigen::Map<Eigen::RowVectorXd>(m_i.data(), m_i.size());
        } else {
            spdlog::error("Unsupported metric dimension: {}", dim);
            exit(1);
        }
    }

    return m;
}

double metric_scale_factor_max(const Eigen::MatrixXd &metric) {
    const int nf = metric.rows();
    assert(metric.cols() == 4 || metric.cols() == 9);
    double max_eigenvalue = 0.0;
    for (int i = 0; i < nf; i++) {
        Eigen::MatrixXd m;
        if (metric.cols() == 4) {
            m.resize(2, 2);
            m << metric(i, 0), metric(i, 1),
                metric(i, 2), metric(i, 3);
        } else {
            m.resize(3, 3);
            m << metric(i, 0), metric(i, 1), metric(i, 2),
                metric(i, 3), metric(i, 4), metric(i, 5),
                metric(i, 6), metric(i, 7), metric(i, 8);
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(m);
        if (eigensolver.info() != Eigen::Success) {
            spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
            exit(1);
        }
        const auto eigenvalues = eigensolver.eigenvalues();
        max_eigenvalue = std::max(max_eigenvalue, eigenvalues.maxCoeff());
    }
    return max_eigenvalue;
}

double metric_scale_factor_avg(const Eigen::MatrixXd &metric) {
    const int nf = metric.rows();
    assert(metric.cols() == 4 || metric.cols() == 9);
    double sum_eig = 0.0;
    for (int i = 0; i < nf; i++) {
        Eigen::MatrixXd m;
        if (metric.cols() == 4) {
            m.resize(2, 2);
            m << metric(i, 0), metric(i, 1),
                metric(i, 2), metric(i, 3);
        } else {
            m.resize(3, 3);
            m << metric(i, 0), metric(i, 1), metric(i, 2),
                metric(i, 3), metric(i, 4), metric(i, 5),
                metric(i, 6), metric(i, 7), metric(i, 8);
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(m);
        if (eigensolver.info() != Eigen::Success) {
            spdlog::error("Failed to compute eigenvalues in {}.", __LINE__);
            exit(1);
        }
        const auto eigenvalues = eigensolver.eigenvalues();
        sum_eig += eigenvalues.mean();
    }
    return sum_eig / nf;
}

Eigen::MatrixXd metric_induced_halfedge_length_3d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &M) {
    const int ne = E.rows();
    const int nv = V.rows();
    assert(E.cols() == 2);
    assert(M.rows() == nv);
    assert(M.cols() == 9);
    Eigen::MatrixXd metric_halfedge_length;
    metric_halfedge_length.resize(ne, 2);

    for (int ei = 0; ei < ne; ei++) {
        const int v1 = E(ei, 0);
        const int v2 = E(ei, 1);
        Eigen::Matrix3d H1, H2;
        H1 << M(v1, 0), M(v1, 1), M(v1, 2),
            M(v1, 3), M(v1, 4), M(v1, 5),
            M(v1, 6), M(v1, 7), M(v1, 8);
        H2 << M(v2, 0), M(v2, 1), M(v2, 2),
            M(v2, 3), M(v2, 4), M(v2, 5),
            M(v2, 6), M(v2, 7), M(v2, 8);

        const Eigen::Vector3d e = V.row(v2) - V.row(v1);
        const double l1 = e.transpose() * H1 * e;
        const double l2 = e.transpose() * H2 * e;
        metric_halfedge_length(ei, 0) = l1 > 0 ? l1 : 0;
        metric_halfedge_length(ei, 1) = l2 > 0 ? l2 : 0;
    }

    return metric_halfedge_length;
}

Eigen::VectorXd metric_induced_edge_length_3d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &metric) {

    Eigen::MatrixXd metric_halfedge_length =
        metric_induced_halfedge_length_3d(V, E, metric);

    Eigen::VectorXd metric_edge_length =
        0.5 * (metric_halfedge_length.col(0) +
               metric_halfedge_length.col(1));
    return metric_edge_length;
}

Eigen::MatrixXd metric_induced_halfedge_length_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXd &B,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &M) {
    const int ne = E.rows();
    const int nv = V.rows();
    assert(E.cols() == 2);
    assert(M.rows() == nv);
    assert(M.cols() == 4);
    Eigen::MatrixXd metric_halfedge_length;
    metric_halfedge_length.resize(ne, 2);

    for (int ei = 0; ei < ne; ei++) {
        const int v1 = E(ei, 0);
        const int v2 = E(ei, 1);
        Eigen::Matrix2d H1, H2;
        H1 << M(v1, 0), M(v1, 1), M(v1, 2), M(v1, 3);
        H2 << M(v2, 0), M(v2, 1), M(v2, 2), M(v2, 3);

        const Eigen::Vector3d edge_3d = V.row(v2) - V.row(v1);
        Eigen::Vector2d edge_2d_v1, edge_2d_v2;
        edge_2d_v1(0) = edge_3d.dot(B.row(v1).head<3>());
        edge_2d_v1(1) = edge_3d.dot(B.row(v1).tail<3>());
        edge_2d_v2(0) = edge_3d.dot(B.row(v2).head<3>());
        edge_2d_v2(1) = edge_3d.dot(B.row(v2).tail<3>());
        const double l1 = edge_2d_v1.transpose() * H1 * edge_2d_v1;
        const double l2 = edge_2d_v2.transpose() * H2 * edge_2d_v2;
        metric_halfedge_length(ei, 0) = l1 > 0 ? l1 : 0;
        metric_halfedge_length(ei, 1) = l2 > 0 ? l2 : 0;
    }

    return metric_halfedge_length;
}

Eigen::VectorXd metric_induced_edge_length_2d(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXd &B,
    const Eigen::MatrixXi &E,
    const Eigen::MatrixXd &metric) {

    Eigen::MatrixXd metric_halfedge_length =
        metric_induced_halfedge_length_2d(V, B, E, metric);

    Eigen::VectorXd metric_edge_length =
        0.5 * (metric_halfedge_length.col(0) +
               metric_halfedge_length.col(1));
    return metric_edge_length;
}

Eigen::Matrix3d metric_projection(
    const Eigen::Matrix3d &M,
    const Eigen::Vector3d &N) {
    Eigen::Matrix3d M_eff = M - (M * N) * (N.transpose() * M) / (N.transpose() * M * N);
    return M_eff;
}

Eigen::MatrixXd metric_projection_batch(
    const Eigen::MatrixXd &M,
    const Eigen::MatrixXd &N) {
    const int nv = M.rows();
    assert(M.rows() == N.rows());
    assert(M.cols() == 9);
    assert(N.cols() == 3);

    Eigen::MatrixXd M_eff;
    M_eff.resize(nv, 9);
    for (int i = 0; i < nv; i++) {
        Eigen::Matrix3d mi;
        mi << M(i, 0), M(i, 1), M(i, 2),
            M(i, 3), M(i, 4), M(i, 5),
            M(i, 6), M(i, 7), M(i, 8);
        Eigen::Vector3d ni = N.row(i);
        Eigen::Matrix3d mi_eff = metric_projection(mi, ni);
        M_eff.row(i) = Eigen::Map<Eigen::RowVectorXd>(mi_eff.data(), mi_eff.size());
    }
    return M_eff;
}

// must follow the metric_projection
Eigen::Matrix2d metric_projection_to_2d(
    const Eigen::Matrix3d &m,
    const Eigen::Vector3d &b1,
    const Eigen::Vector3d &b2) {
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = b1;
    basis.col(1) = b2;

    Eigen::Matrix2d m_2d = basis.transpose() * m * basis;
    return m_2d;
}

Eigen::MatrixXd metric_projection_to_2d_batch(
    const Eigen::MatrixXd &M,
    const Eigen::MatrixXd &B) {
    const int nv = M.rows();
    assert(M.cols() == 9);
    assert(B.cols() == 6);
    assert(B.rows() == nv);

    Eigen::MatrixXd M_eff;
    M_eff.resize(nv, 4);
    for (int i = 0; i < nv; i++) {
        Eigen::Matrix3d mi;
        mi << M(i, 0), M(i, 1), M(i, 2),
            M(i, 3), M(i, 4), M(i, 5),
            M(i, 6), M(i, 7), M(i, 8);
        Eigen::Vector3d b1 = B.row(i).head(3);
        Eigen::Vector3d b2 = B.row(i).tail(3);
        Eigen::Matrix2d mi_eff = metric_projection_to_2d(mi, b1, b2);
        M_eff.row(i) = Eigen::Map<Eigen::RowVectorXd>(mi_eff.data(), mi_eff.size());
    }
    return M_eff;
}

} // namespace zxh::metric_utils