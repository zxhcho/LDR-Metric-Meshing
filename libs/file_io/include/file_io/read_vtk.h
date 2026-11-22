#pragma once

#include "vtk_property.h"

#include <Eigen/Dense>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace zxh::file_io {

int read_vtk(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    std::vector<std::vector<size_t>> &C);

int read_vtk(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &C);

template <typename Property>
int read_vtk_with_attribute(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &C,
    Property &P) {

    using namespace std;
    ifstream ifs(file_name);
    if (ifs.fail()) {
        std::cout << "can not open file " << file_name << std::endl;
        return __LINE__;
    }

    string str;
    while (!ifs.eof()) {
        ifs >> str;
        if (str == "POINTS") {
            size_t point_num;
            ifs >> point_num >> str;
            if (point_num > 0) {
                V.resize(point_num, 3);
                for (size_t i = 0; i < point_num; ++i) {
                    for (size_t j = 0; j < 3; ++j)
                        ifs >> V(i, j);
                }
            }
        } else if (str == "CELLS") {
            size_t cell_num, total_num, vertices_per_cell, tmp;
            ifs >> cell_num >> total_num;
            if (cell_num > 0) {
                vertices_per_cell = total_num / cell_num - 1;
                C.resize(cell_num, vertices_per_cell);
                for (size_t ci = 0; ci < cell_num; ++ci) {
                    ifs >> tmp;
                    assert(tmp == vertices_per_cell);
                    for (size_t j = 0; j < vertices_per_cell; ++j) {
                        ifs >> C(ci, j);
                    }
                }
            }
        } else if (str == "CELL_DATA" || str == "POINT_DATA") {
            int n;
            ifs >> n;
            if (n > 0) {
                P.resize(n);
                for (int i = 0; i < 6; i++) {
                    ifs >> str;
                }
                for (int i = 0; i < n; ++i) {
                    ifs >> P[i];
                }
            }
            break;
        }
    }
    ifs.close();

    return 0;
}

int read_vtk_with_attribute(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &C,
    std::vector<vtk_property> &properties);

} // namespace zxh::file_io