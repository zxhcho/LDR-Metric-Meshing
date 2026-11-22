#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#include <spdlog/spdlog.h>

#include <igl/readOBJ.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/boundary_facets.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/edge_topology.h>
#include <igl/avg_edge_length.h>

#include <file_io/read_vtk.h>
#include <file_io/write_vtk.h>
#include <file_io/mat_io.h>

#include <metric_utils/metric_op.h>

#include <string>

using namespace std;
using namespace Eigen;

using namespace zxh::file_io;
using namespace zxh::metric_utils;

int main(int argc, char **argv) {
    CLI::App app{"Discrete metric from quadric field fitting"};
    string field_filee, metric_file;
    int k_ring = 3, dim = 2;
    app.add_option("--field", field_filee)->required();
    app.add_option("--metric", metric_file, "metric output file");
    app.add_option("--dim", dim, "dimension of the mesh")->default_val(2);
    app.add_option("--k", k_ring, "Ring size for metric computation")->default_val(3);
    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::VectorXd field;
    read_vtk_with_attribute(field_filee, V, F, field);
    const int nv = V.rows();
    const int nf = F.rows();

    Eigen::MatrixXd metric;
    if (dim == 2) {
        metric = metric_from_hessian_2d(V, F, field, k_ring);

    } else if (dim == 3) {
        metric = metric_from_hessian_3d(V, F, field, k_ring);
    }
    metric_spd(metric);
    if (dim == 3) {
        Eigen::MatrixXd N;
        igl::per_vertex_normals(V, F, N);
        metric = metric_projection_batch(metric, N);
    }

    const double scale = metric_scale_factor_max(metric);
    metric /= scale;

    if (dim == 2) {
        metric = zxh::metric_utils::metric_2d_to_3d(metric);
    }

    write_dense_mat(metric_file, metric);

    return 0;
}