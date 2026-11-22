#include "file_io/point_io.h"

#include <string>
#include <iostream>
#include <fstream>

#include <vector>

namespace zxh::file_io {

using namespace std;
using namespace Eigen;

int read_xyz(
    const std::string &file_name,
    Eigen::MatrixXd &V) {

    std::ifstream ifs(file_name);
    if (ifs.fail()) {
        std::cout << "can not open file " << file_name << std::endl;
        return __LINE__;
    }

    std::vector<double> data;
    double val;
    while (!ifs.eof()) {
        ifs >> val;
        data.push_back(val);
    }
    const int point_num = data.size() / 3;
    std::cout << "read " << point_num << " points" << std::endl;
    V.resize(point_num, 3);
    for (int i = 0; i < point_num; ++i) {
        for (int j = 0; j < 3; ++j) {
            V(i, j) = data[i * 3 + j];
        }
    }

    return 0;
}

} // namespace zxh::file_io