#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#include <spdlog/spdlog.h>

#include <igl/readOBJ.h>

#include <pde/project.h>

#include <Eigen/CholmodSupport>

#include <file_io/read_vtk.h>
#include <file_io/write_vtk.h>

#include <string>

using namespace std;
using namespace Eigen;

using namespace zxh::file_io;
using namespace zxh::pde;

int main(int argc, char **argv) {
    CLI::App app{"Solve homogenous Poisson equation from given discrete function"};
    string obj_file, field_ref_file, field_project_file;
    app.add_option("--obj", obj_file)->required();
    app.add_option("--field", field_ref_file)->required();
    app.add_option("--project", field_project_file)->required();
    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(obj_file, V, F);
    spdlog::debug("Read obj file {} with {} vertices and {} faces", obj_file, V.rows(), F.rows());

    Eigen::MatrixXd V_ref;
    Eigen::MatrixXi F_ref;
    Eigen::VectorXd field;
    read_vtk_with_attribute(field_ref_file, V_ref, F_ref, field);
    spdlog::debug("Read reference file {} with {} vertices and {} faces", field_ref_file, V_ref.rows(), F_ref.rows());

    Eigen::VectorXi I;
    Eigen::MatrixXd C, BC;
    project_onto_mesh(
        V, V_ref, F_ref, I, C, BC);

    Eigen::VectorXd field_proj;
    project_scalar_field_onto_mesh(
        field, F_ref, I, BC, field_proj);

    write_vtk_with_attribute(field_project_file, V, F, field_proj);
}