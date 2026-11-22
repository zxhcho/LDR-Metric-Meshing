#include "file_io/read_vtk.h"

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace zxh::file_io {

using namespace std;
using namespace Eigen;

//* read hybrid mesh vtk
//* V should be pure data type instead of an expression
int read_vtk(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    std::vector<std::vector<size_t>> &C) {

    std::ifstream ifs(file_name);
    if (ifs.fail()) {
        std::cerr << "can not open file " << file_name << std::endl;
        return __LINE__;
    }

    std::string str;
    while (!ifs.eof()) {
        ifs >> str;
        if (str == "POINTS") {
            size_t point_num;
            ifs >> point_num >> str;
            V.resize(point_num, 3);
            for (size_t i = 0; i < point_num; ++i) {
                for (size_t j = 0; j < 3; ++j)
                    ifs >> V(i, j);
            }
        } else if (str == "CELLS") {
            size_t cell_num, total_num, vertices_per_cell, tmp;
            ifs >> cell_num >> total_num;
            C.resize(cell_num);
            for (size_t ci = 0; ci < cell_num; ++ci) {
                ifs >> vertices_per_cell;
                C[ci].resize(vertices_per_cell);
                for (size_t vi = 0; vi < vertices_per_cell; ++vi) {
                    ifs >> C[ci][vi];
                }
            }
            break;
        }
    }
    ifs.close();

    return 0;
}

//* read pure mesh vtk
//* V,C should be pure data type instead of an expression
int read_vtk(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &C) {
    std::ifstream ifs(file_name);
    if (ifs.fail()) {
        std::cerr << "can not open file " << file_name << std::endl;
        return __LINE__;
    }

    std::string str;
    while (!ifs.eof()) {
        ifs >> str;
        if (str == "POINTS") {
            size_t point_num;
            ifs >> point_num >> str;
            V.resize(point_num, 3);
            for (size_t i = 0; i < point_num; ++i) {
                for (size_t j = 0; j < 3; ++j)
                    ifs >> V(i, j);
            }
        } else if (str == "CELLS") {
            size_t cell_num, total_num, vertices_per_cell, tmp;
            ifs >> cell_num >> total_num;
            vertices_per_cell = total_num / cell_num - 1;
            C.resize(cell_num, vertices_per_cell);
            for (size_t ci = 0; ci < cell_num; ++ci) {
                ifs >> tmp;
                assert(tmp == vertices_per_cell);
                for (size_t j = 0; j < vertices_per_cell; ++j) {
                    ifs >> C(ci, j);
                }
            }
            break;
        }
    }
    ifs.close();

    return 0;
}

//* read vtk with single scalar attributes
//* V,C,P should be pure data type instead of an expression

// int read_vtk_with_attribute(
//     const std::string &file_name,
//     Eigen::MatrixXd &V,
//     Eigen::MatrixXi &C,
//     Eigen::VectorXd &P) {
//     using namespace std;
//     ifstream ifs(file_name);
//     if (ifs.fail()) {
//         std::cerr << "can not open file " << file_name << std::endl;
//         return __LINE__;
//     }

//     string str;
//     while (!ifs.eof()) {
//         ifs >> str;
//         if (str == "POINTS") {
//             size_t point_num;
//             ifs >> point_num >> str;
//             V.resize(point_num, 3);
//             for (size_t i = 0; i < point_num; ++i) {
//                 for (size_t j = 0; j < 3; ++j)
//                     ifs >> V(i, j);
//             }
//         } else if (str == "CELLS") {
//             size_t cell_num, total_num, vertices_per_cell, tmp;
//             ifs >> cell_num >> total_num;
//             vertices_per_cell = total_num / cell_num - 1;
//             C.resize(cell_num, vertices_per_cell);
//             for (size_t ci = 0; ci < cell_num; ++ci) {
//                 ifs >> tmp;
//                 assert(tmp == vertices_per_cell);
//                 for (size_t j = 0; j < vertices_per_cell; ++j) {
//                     ifs >> C(ci, j);
//                 }
//             }
//         } else if (str == "CELL_DATA" || str == "POINT_DATA") {
//             int n;
//             ifs >> n;
//             P.resize(n);
//             for (int i = 0; i < 6; i++) {
//                 ifs >> str;
//             }
//             for (size_t i = 0; i < n; ++i) {
//                 ifs >> P[i];
//             }
//             break;
//         }
//     }
//     ifs.close();

//     return 0;
// }

int read_vtk_with_attribute(
    const std::string &file_name,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &C,
    std::vector<vtk_property> &properties) {
    using namespace std;
    ifstream ifs(file_name);
    if (ifs.fail()) {
        std::cerr << "can not open file " << file_name << std::endl;
        return __LINE__;
    }

    string str;
    while (!ifs.eof()) {
        ifs >> str;
        if (str == "POINTS") {
            size_t point_num;
            ifs >> point_num >> str;
            V.resize(point_num, 3);
            for (size_t i = 0; i < point_num; ++i) {
                for (size_t j = 0; j < 3; ++j)
                    ifs >> V(i, j);
            }
        } else if (str == "CELLS") {
            size_t cell_num, total_num, vertices_per_cell, tmp;
            ifs >> cell_num >> total_num;
            vertices_per_cell = total_num / cell_num - 1;
            C.resize(cell_num, vertices_per_cell);
            for (size_t ci = 0; ci < cell_num; ++ci) {
                ifs >> tmp;
                assert(tmp == vertices_per_cell);
                for (size_t j = 0; j < vertices_per_cell; ++j) {
                    ifs >> C(ci, j);
                }
            }
        } else if (str == "CELL_DATA" || str == "POINT_DATA") {
            int n;
            ifs >> n;
            while (!ifs.eof()) {
                string str;
                ifs >> str;
                if (str == "CELL_DATA" || str == "POINT_DATA") {
                    break;
                }
                vtk_property p;
                p.data.resize(n);
                ifs >> p.name;
                for (int i = 0; i < 4; i++) {
                    ifs >> str;
                }
                for (int i = 0; i < n; i++) {
                    ifs >> p.data[i];
                }
                properties.push_back(p);
            }
        }
    }
    ifs.close();

    return 0;
}

} // namespace zxh::file_io