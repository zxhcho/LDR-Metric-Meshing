#include <pde/laplacian.h>

namespace zxh::pde {

int squared_edge_length(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::MatrixXd &E) {
    const int nf = F.rows();
    E.resize(nf, 3);
    for (int fi = 0; fi < nf; fi++) {
        // expand flattened metric tensor
        const Eigen::VectorXd mi_vec = M.row(fi);
        const Eigen::Matrix3d mi = Eigen::Map<const Eigen::Matrix3d>(mi_vec.data(), 3, 3);
        const Eigen::Vector3d e0 = V.row(F(fi, 1)) - V.row(F(fi, 2));
        const Eigen::Vector3d e1 = V.row(F(fi, 2)) - V.row(F(fi, 0));
        const Eigen::Vector3d e2 = V.row(F(fi, 0)) - V.row(F(fi, 1));
        E(fi, 0) = e0.transpose() * mi * e0;
        E(fi, 1) = e1.transpose() * mi * e1;
        E(fi, 2) = e2.transpose() * mi * e2;
    }
    return 0;
}

int area_from_squared_edge_length(
    const Eigen::MatrixXd &l2,
    Eigen::VectorXd &A) {
    const int nf = l2.rows();
    A.resize(nf);
    for (int fi = 0; fi < nf; fi++) {
        const double a2 = l2(fi, 0);
        const double b2 = l2(fi, 1);
        const double c2 = l2(fi, 2);
        const double area = 0.25 * sqrt(4 * a2 * b2 - (a2 + b2 - c2) * (a2 + b2 - c2));
        A(fi) = std::max(area, 1e-10);
    }
    return 0;
}

int compute_cotangent_entries(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::MatrixXd &C) {
    const int nf = F.rows();
    C.resize(nf, 3);

    Eigen::MatrixXd l2;
    l2.resize(nf, 3);
    squared_edge_length(V, F, M, l2);

    Eigen::VectorXd A;
    area_from_squared_edge_length(l2, A);

    for (int fi = 0; fi < nf; fi++) {
        const double a2 = l2(fi, 0);
        const double b2 = l2(fi, 1);
        const double c2 = l2(fi, 2);
        C(fi, 0) = (b2 + c2 - a2) / (8.0 * A(fi));
        C(fi, 1) = (c2 + a2 - b2) / (8.0 * A(fi));
        C(fi, 2) = (a2 + b2 - c2) / (8.0 * A(fi));
    }

    return 0;
}

int compute_laplacian_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::SparseMatrix<double> &L) {

    const int nf = F.rows();
    const int nv = V.rows();
    Eigen::MatrixXd C;
    compute_cotangent_entries(V, F, M, C);
    L.reserve(10 * nv);
    L.resize(nv, nv);
    Eigen::Matrix<int, 3, 2> edges;
    edges << 1, 2,
        2, 0,
        0, 1;

    std::vector<Eigen::Triplet<double>> IJV;
    IJV.reserve(nf * 12);
    for (int fi = 0; fi < nf; fi++) {
        for (int e = 0; e < edges.rows(); e++) {
            const int source = F(fi, edges(e, 0));
            const int dest = F(fi, edges(e, 1));
            IJV.push_back(Eigen::Triplet<double>(source, dest, C(fi, e)));
            IJV.push_back(Eigen::Triplet<double>(dest, source, C(fi, e)));
            IJV.push_back(Eigen::Triplet<double>(source, source, -C(fi, e)));
            IJV.push_back(Eigen::Triplet<double>(dest, dest, -C(fi, e)));
        }
    }

    L.setFromTriplets(IJV.begin(), IJV.end());

    return 0;
}

int compute_mass_matrix(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &M,
    Eigen::SparseMatrix<double> &Mass) {
    const int nf = F.rows();
    const int nv = V.rows();

    Eigen::MatrixXd l2;
    l2.resize(nf, 3);
    squared_edge_length(V, F, M, l2);

    Eigen::VectorXd A;
    area_from_squared_edge_length(l2, A);

    return 0;
}
} // namespace zxh::pde