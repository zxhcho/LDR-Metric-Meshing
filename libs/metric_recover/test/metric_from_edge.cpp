#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>
#include <spdlog/spdlog.h>

#include <string>

#include <igl/readOBJ.h>
#include <igl/edges.h>

#include <file_io/mat_io.h>
#include <file_io/read_vtk.h>

#include <metric_recover/mesh/mesh.h>
#include <metric_recover/problem/vertex_metric.h>

#include <metric_utils/metric_op.h>

using namespace std;
using namespace zxh::file_io;
using namespace zxh::metric_utils;
using namespace zxh::metric_recover;

int main(int argc, char **argv) {
    CLI::App app{"Metric Field Recovery from Edge Lengths"};
    string obj_file, metric_file, edge_file;
    int dim, dual;
    app.add_option("--obj", obj_file, "input obj file");
    app.add_option("--edge", edge_file, "input halfedge/edge length file");
    app.add_option("--metric", metric_file, "output recovered metric file");
    app.add_option("--dim", dim, "dimension of the problem")->required();
    app.add_option("--dual", dual, "use dual cells")->default_val(0);
    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(obj_file, V, F)) {
        spdlog::error("Failed to read obj file: {}", obj_file);
        return 1;
    }

    Eigen::MatrixXi EV;
    Eigen::VectorXd edge_lens;
    zxh::file_io::read_vtk_with_attribute(edge_file, V, EV, edge_lens);

    //* Important: the edge loaded must be the same as igl::edge_topology
    const mesh_t mesh = compute_mesh(V, F);

    Eigen::MatrixXd metric;
    if (dim == 2) {
        if (dual == 0) {
            vertex_metric_solver_edge(mesh, edge_lens, metric);
        } else if (dual == 1) {
            vertex_metric_solver_dual_edge(mesh, edge_lens, metric);
        }
        metric = metric_2d_to_3d(metric);
    } else if (dim == 3) {
        if (dual == 0)
            vertex_metric_solver_edge_tangent(mesh, edge_lens, metric);
        else if (dual == 1)
            vertex_metric_solver_dual_edge_tangent(mesh, edge_lens, metric);
    }

    if (write_dense_mat(metric_file, metric)) {
        spdlog::error("Failed to write metric file: {}", metric_file);
        return 1;
    }
}