#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>
#include <spdlog/spdlog.h>

#include <string>

#include <Eigen/Dense>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/avg_edge_length.h>

#include <file_io/mat_io.h>

#include <mesh_adapter/mmg_adapter.h>
#include <mesh_adapter/metric_normalize.h>

using namespace std;
using namespace zxh::file_io;

int main(int argc, char **argv) {
    CLI::App app{"Given a mesh and a metric field, adapt the mesh using MMG library."};
    string in_file, out_file, metric_file;
    double hmin, hmax;
    int dof;
    int isotropic = 0;
    app.add_option("--in", in_file);
    app.add_option("--out", out_file);
    app.add_option("--metric", metric_file);
    app.add_option("--hmin", hmin)->default_val(0.01);
    app.add_option("--hmax", hmax)->default_val(0.1);
    app.add_option("--dof", dof)->default_val(-1);
    app.add_option("--isotropic", isotropic)->default_val(0);
    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(in_file, V, F);
    const int nv = V.rows();
    const int nf = F.rows();
    spdlog::info("Input mesh: {} vertices, {} faces.", nv, nf);

    const double avg_edge_len = igl::avg_edge_length(V, F);
    hmin *= avg_edge_len;
    hmax *= avg_edge_len;

    Eigen::MatrixXd metric;
    if (read_dense_mat(metric_file, metric)) {
        spdlog::error("Failed to read metric file {}", metric_file);
        return __LINE__;
    }
    spdlog::info("Mesh size: {},{}", metric.rows(), metric.cols());

    spdlog::info("Begin Norm");

    if (dof != -1) {
        spdlog::info("Target DOF: {}", dof);
        if (isotropic)
            zxh::mesh_adapter::metric_normalize_isotropic(V, F, metric, dof, hmin, hmax);
        else
            zxh::mesh_adapter::metric_normalize(V, F, metric, dof, hmin, hmax);
    }

    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;

    spdlog::info("Begin Adapt");

    if (zxh::mesh_adapter::adapt_mmg_2d(V, F, metric, V_out, F_out, hmin, hmax)) {
        spdlog::error("Mesh adaption failed.");
        return __LINE__;
    }

    spdlog::info("End");

    spdlog::info("Mesh adaption succeeded. Output mesh: {} vertices, {} faces.", V_out.rows(), F_out.rows());
    igl::writeOBJ(out_file, V_out, F_out);
}