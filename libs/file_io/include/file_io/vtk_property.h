#pragma once

#include <Eigen/Dense>

#include <string>

namespace zxh::file_io {

struct vtk_property {
public:
    Eigen::VectorXd data;
    std::string name;
};

} // namespace zxh::file_io