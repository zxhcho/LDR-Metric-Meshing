#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#include <yaml-cpp/yaml.h>

#include <spdlog/spdlog.h>

#include <igl/doublearea.h>
#include <igl/grad.h>

#include <file_io/read_vtk.h>
#include <file_io/write_vtk.h>

#include <pde/integral_numerical.h>

#include <string>

using namespace std;
using namespace Eigen;

using namespace zxh::file_io;
using namespace zxh::pde;

int main(int argc, char **argv) {
    CLI::App app{"Compare the Integral of C0 error between numerical results in sparse mesh and ground truth defined in dense mesh"};
    string gt_file, solve_file, log_file, error_file;
    int order = 0;
    app.add_option("--solve", solve_file)->required();
    app.add_option("--gt", gt_file)->required();
    app.add_option("--log", log_file);
    app.add_option("--error", error_file);
    app.add_option("--order", order, "order of the error")->default_val(0);
    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd V_solve;
    Eigen::MatrixXi F_solve;
    Eigen::VectorXd Solve;
    read_vtk_with_attribute(solve_file, V_solve, F_solve, Solve);
    const int nv_solve = V_solve.rows();
    const int nf_solve = F_solve.rows();
    spdlog::debug("Read solve file {} with {} vertices and {} faces", solve_file, nv_solve, nf_solve);

    Eigen::MatrixXd V_gt;
    Eigen::MatrixXi F_gt;
    Eigen::VectorXd GT;
    read_vtk_with_attribute(gt_file, V_gt, F_gt, GT);
    const int nv_gt = V_gt.rows();
    const int nf_gt = F_gt.rows();
    spdlog::debug("Read gt file {} with {} vertices and {} faces", gt_file, nv_gt, nf_gt);

    Eigen::VectorXd A_gt;
    igl::doublearea(V_gt, F_gt, A_gt);

    // Resolution of discrete ground truth should be higher than that of numerical solution
    assert(nv_solve <= nv_gt);
    assert(nf_solve <= nf_gt);

    YAML::Node config;
    //* for scalar error
    {
        Eigen::VectorXd c0_jacs_l1, c0_jacs_l2;
        triangle_integral_jac(
            V_gt, F_gt, GT,
            c0_jacs_l1, c0_jacs_l2);
        const double c0_l1 = c0_jacs_l1.dot(A_gt);
        const double c0_l2 = sqrt(c0_jacs_l2.dot(A_gt));

        Eigen::VectorXd c0_error_jacs_l1, c0_error_jacs_l2;
        triangle_integral_diff_jac(
            V_gt, F_gt, GT,
            V_solve, F_solve, Solve,
            c0_error_jacs_l1, c0_error_jacs_l2);

        const double c0_error_l1 = c0_error_jacs_l1.dot(A_gt);
        const double c0_error_l2 = sqrt(c0_error_jacs_l2.dot(A_gt));

        const double c0_rel_error_l1 = c0_error_l1 / c0_l1;
        const double c0_rel_error_l2 = c0_error_l2 / c0_l2;

        config["DOF"] = nv_solve;
        config["c0_l1"] = c0_l1;
        config["c0_l2"] = c0_l2;
        config["c0_abs_error_l1"] = c0_error_l1;
        config["c0_abs_error_l2"] = c0_error_l2;
        config["c0_rel_error_l1"] = c0_rel_error_l1;
        config["c0_rel_error_l2"] = c0_rel_error_l2;

        if (!error_file.empty()) {
            write_vtk_with_attribute(error_file + "_c0_jac.vtk", V_gt, F_gt, c0_error_jacs_l2.cwiseSqrt(), "c0_jac_l2");
        }
    }
    if (order == 1) {
        //* calculate the gradient of the sparse and dense mesh
        Eigen::SparseMatrix<double> G_solve, G_gt;
        igl::grad(V_solve, F_solve, G_solve);
        igl::grad(V_gt, F_gt, G_gt);
        MatrixXd grad_solve = Eigen::Map<const MatrixXd>(
            (G_solve * Solve).eval().data(), nf_solve, 3);
        for (int i = 0; i < grad_solve.rows(); ++i) {
            const double norm = grad_solve.row(i).norm();
            if (isnan(norm)) {
                spdlog::warn("NaN in solve grad {}", i);
                spdlog::warn("solve grad {},{},{}",
                             grad_solve(i, 0), grad_solve(i, 1), grad_solve(i, 2));
                grad_solve.row(i) = Eigen::Vector3d::Zero();
            }
        }
        const MatrixXd grad_gt = Eigen::Map<const MatrixXd>(
            (G_gt * GT).eval().data(), nf_gt, 3);

        spdlog::debug("Solve grad norm: {}", grad_solve.rowwise().norm().mean());
        spdlog::debug("GT grad norm: {}", grad_gt.rowwise().norm().mean());

        Eigen::VectorXd c1_jacs_l1, c1_jacs_l2;
        triangle_integral_jac(
            V_gt, F_gt, grad_gt,
            c1_jacs_l1, c1_jacs_l2);
        const double c1_l1 = c1_jacs_l1.dot(A_gt);
        const double c1_l2 = sqrt(c1_jacs_l2.dot(A_gt));

        Eigen::VectorXd c1_error_jacs_l1, c1_error_jacs_l2;
        triangle_integral_diff_jac(
            V_gt, F_gt, grad_gt,
            V_solve, F_solve, grad_solve,
            c1_error_jacs_l1, c1_error_jacs_l2);

        const double c1_error_l1 = c1_error_jacs_l1.dot(A_gt);
        const double c1_error_l2 = sqrt(c1_error_jacs_l2.dot(A_gt));

        const double c1_rel_error_l1 = c1_error_l1 / c1_l1;
        const double c1_rel_error_l2 = c1_error_l2 / c1_l2;

        config["c1_l1"] = c1_l1;
        config["c1_l2"] = c1_l2;
        config["c1_abs_error_l1"] = c1_error_l1;
        config["c1_abs_error_l2"] = c1_error_l2;
        config["c1_rel_error_l1"] = c1_rel_error_l1;
        config["c1_rel_error_l2"] = c1_rel_error_l2;

        if (!error_file.empty()) {
            write_vtk_with_attribute(error_file + "_c1_jac.vtk", V_gt, F_gt, c1_error_jacs_l2.cwiseSqrt(), "c1_jac_l2");
        }
    }

    std::ofstream fout(log_file);
    fout << config;

    return 0;
}