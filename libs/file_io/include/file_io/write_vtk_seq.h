#pragma once

#include "cell_types.h"
#include "vtk_property.h"
#include "write_vtk.h"

#include <iostream>
#include <fstream>
#include <filesystem>

#include <Eigen/Dense>

namespace zxh::file_io {

//* record a sequence of vtk file in a diractory:
//* pattern: f"{out_dir}/{name}/{name}_{n_iter}.vtk"
template <typename DerivedV,
          typename DerivedF>
int record_vtk_seq(
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    const std::filesystem::path &out_dir,
    const std::string &name,
    const size_t n_iter,
    const size_t n_zero = 4) {
    const std::filesystem::path sub_dir = out_dir / name;
    if (!std::filesystem::exists(sub_dir)) //* create a new diractory at first time
        if (!std::filesystem::create_directory(sub_dir)) {
            std::cout << "create dir " << sub_dir << "failed" << std::endl;
        }

    std::string num_str = std::string(n_zero - std::min(n_zero, std::to_string(n_iter).length()), '0') + std::to_string(n_iter);

    const std::string file_name = (sub_dir / (name + "_" + num_str + ".vtk")).string();
    write_vtk(file_name, V, F);

    return 0;
}

//* record a sequence of vtk file with property in a diractory:
//* pattern: f"{out_dir}/{name}/{name}_{n_iter}.vtk"
template <typename DerivedV,
          typename DerivedF,
          typename DerivedX>
int record_property_vtk_seq(
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    const Eigen::MatrixBase<DerivedX> &property,
    const std::filesystem::path &out_dir,
    const std::string &name,
    const size_t n_iter,
    const size_t n_zero = 4) {

    const std::filesystem::path sub_dir = out_dir / name;

    std::string num_str = std::string(n_zero - std::min(n_zero, std::to_string(n_iter).length()), '0') + std::to_string(n_iter);

    if (!std::filesystem::exists(sub_dir)) //* create a new diractory at first time
        if (!std::filesystem::create_directory(sub_dir)) {
            std::cout << "create dir " << sub_dir << "failed" << std::endl;
        }
    const std::string file_name = (sub_dir / (name + "_" + num_str + ".vtk")).string();

    write_vtk_with_attribute(file_name, V, F, property, name);

    return 0;
}

} // namespace zxh::file_io