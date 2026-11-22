#pragma once

#include <Eigen/Dense>

#include <string>
#include <fstream>
#include <iostream>

namespace zxh::file_io {

int read_xyz(const std::string &file_name,
             Eigen::MatrixXd &V);

template <typename DerivedV>
int write_xyz(const std::string &file_name,
              const Eigen::MatrixBase<DerivedV> &V) {
    std::ofstream ofs(file_name);
    if (ofs.fail()) {
        std::cout << "can not open file " << file_name << std::endl;
        return __LINE__;
    }
    const int point_num = V.rows();

    for (int i = 0; i < point_num; ++i) {
        for (int j = 0; j < 3; ++j) {
            ofs << V(i, j) << ' ';
        }
        ofs << std::endl;
    }
    return 0;
}

} // namespace zxh::file_io