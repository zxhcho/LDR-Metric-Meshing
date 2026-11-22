#include <mesh_adapter/metric_normalize.h>

#include <Eigen/Dense>

#include <igl/massmatrix.h>

namespace zxh::mesh_adapter {

int metric_normalize(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &metrics,
    const int dof,
    const double hmin,
    const double hmax) {

    Eigen::SparseMatrix<double> M;
    // Calculate the Mass Matrix with Barycentric Lumping
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

    // The Dual Areas are simply the diagonal entries
    Eigen::VectorXd dual_areas = M.diagonal();

    const int nv = metrics.rows();

    // Temporary storage to avoid re-decomposing matrices twice
    Eigen::MatrixXd stored_evals, stored_evecs;
    stored_evals.resize(nv, 2);
    stored_evecs.resize(nv, 4); // Each row stores

    // --- STEP 1: Compute Total Complexity ---
    // Complexity = Integral( sqrt(det(M)) ) dx
    // This represents the "ideal" number of vertices the current field asks for.
    double total_complexity = 0.0;

    for (int i = 0; i < nv; i++) {
        // 1. Reconstruct 2x2 Matrix from flattened row
        Eigen::Matrix2d M;
        M << metrics(i, 0), metrics(i, 1),
            metrics(i, 3), metrics(i, 4);

        // 2. Eigen Decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(M);

        // Take absolute values (Hessian can be negative definite, Metric must be positive)
        Eigen::Vector2d evals = es.eigenvalues().cwiseAbs();
        Eigen::Matrix2d evecs = es.eigenvectors();

        stored_evals.row(i) = evals;
        stored_evecs.row(i) = Eigen::Map<Eigen::RowVector4d>(evecs.data(), 4);

        // 3. Accumulate Complexity
        // Density = sqrt(det(M)) = sqrt(lambda1 * lambda2)
        double density = std::sqrt(evals[0] * evals[1]);
        total_complexity += density * dual_areas(i);
    }

    // --- STEP 2: Calculate Global Scaling Factor ---
    // We want the new complexity to equal target_dof.
    // Scaling metric M by S scales density by S (in 2D).
    // So: new_complexity = S * old_complexity = target_dof
    if (total_complexity < 1e-12)
        total_complexity = 1.0; // Prevent div by zero
    double S = (double)dof / total_complexity;

    // --- STEP 3: Apply Scaling & Clamping ---
    // Convert physical length limits to eigenvalue limits
    // lambda = 1 / h^2
    double lambda_max = 1.0 / (hmin * hmin); // Limit for finest elements
    double lambda_min = 1.0 / (hmax * hmax); // Limit for coarsest elements

    for (int i = 0; i < nv; i++) {
        Eigen::Vector2d evals = stored_evals.row(i);
        Eigen::Matrix2d evecs;
        evecs << stored_evecs(i, 0), stored_evecs(i, 1),
            stored_evecs(i, 2), stored_evecs(i, 3);

        // A. Apply Global Scaling
        evals *= S;

        // B. Apply Local Clamping (The "Privilege" Step)
        // This prevents the solution metric from requesting h < h_min
        evals[0] = std::min(std::max(evals[0], lambda_min), lambda_max);
        evals[1] = std::min(std::max(evals[1], lambda_min), lambda_max);

        // C. Reconstruct Metric Tensor
        // M_new = V * Lambda_clamped * V^T
        Eigen::Matrix2d M_new = evecs * evals.asDiagonal() * evecs.transpose();

        // D. Flatten back to output
        metrics(i, 0) = M_new(0, 0);
        metrics(i, 1) = M_new(0, 1);
        metrics(i, 3) = M_new(1, 0);
        metrics(i, 4) = M_new(1, 1);
    }

    return 0;
}

int metric_normalize_isotropic(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &metrics,
    const int dof,
    const double hmin,
    const double hmax) {

    Eigen::SparseMatrix<double> M_mass;
    // Calculate the Mass Matrix with Barycentric Lumping
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M_mass);

    // The Dual Areas are simply the diagonal entries
    Eigen::VectorXd dual_areas = M_mass.diagonal();

    const int nv = metrics.rows();

    // Temporary storage for eigenvalues
    Eigen::MatrixXd stored_evals;
    stored_evals.resize(nv, 2);

    // --- STEP 1: Compute Total Complexity (ISOTROPIC BASIS) ---
    double total_complexity = 0.0;

    for (int i = 0; i < nv; i++) {
        // 1. Reconstruct 2x2 Matrix
        Eigen::Matrix2d M;
        M << metrics(i, 0), metrics(i, 1),
            metrics(i, 3), metrics(i, 4);

        // 2. Eigen Decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(M, Eigen::EigenvaluesOnly);
        Eigen::Vector2d evals = es.eigenvalues().cwiseAbs();
        stored_evals.row(i) = evals;

        // 3. Accumulate Complexity using ISOTROPIC Density
        // For an isotropic metric M = lambda * I, the density is sqrt(det(M)).
        // det(M) = lambda * lambda  ->  sqrt(det) = lambda.
        // We choose lambda = max(evals) to represent the tightest constraint requested.
        double lambda_iso = std::max(evals[0], evals[1]);

        // Complexity contribution = Density * Area
        total_complexity += lambda_iso * dual_areas(i);
    }

    // --- STEP 2: Calculate Global Scaling Factor ---
    // This ensures the final isotropic mesh has exactly 'dof' vertices
    if (total_complexity < 1e-12)
        total_complexity = 1.0;
    double S = (double)dof / total_complexity;

    // --- STEP 3: Apply Scaling & Isotropic Conversion ---
    // Convert physical length limits to eigenvalue limits (1/h^2)
    double lambda_max_limit = 1.0 / (hmin * hmin);
    double lambda_min_limit = 1.0 / (hmax * hmax);

    for (int i = 0; i < nv; i++) {
        Eigen::Vector2d evals = stored_evals.row(i);

        // A. Apply Global Scaling
        evals *= S;

        // B. Determine Local Isotropic Value
        // We take the max eigenvalue (tightest direction)
        double lambda_iso = std::max(evals[0], evals[1]);

        // C. Apply Local Clamping (Privilege Step)
        lambda_iso = std::min(std::max(lambda_iso, lambda_min_limit), lambda_max_limit);

        // D. Write Output as Isotropic Tensor (lambda * Identity)
        metrics(i, 0) = lambda_iso;
        metrics(i, 1) = 0.0;
        metrics(i, 3) = 0.0;
        metrics(i, 4) = lambda_iso;
    }

    return 0;
}

} // namespace zxh::mesh_adapter