#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <string>

#include <iostream>
#include <fstream>

namespace zxh::file_io {

template <typename Mat>
int write_dense_mat(const std::string &out_file,
                    const Mat &mat) {
    using namespace std;
    ofstream ofs(out_file);
    if (!ofs) {
        cerr << "matrix file creat error: " << out_file << endl;
        return __LINE__;
    }

    ofs << mat.rows() << ' ' << mat.cols() << endl;

    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            ofs << mat(i, j) << ' ';
        }
        ofs << endl;
    }

    ofs.close();

    return 0;
}

template <typename Mat>
int read_dense_mat(const std::string &in_file,
                   Mat &mat) {
    using namespace std;
    ifstream ifs(in_file);
    if (!ifs) {
        cerr << "matrix file read error: " << in_file << endl;
        return __LINE__;
    }

    int rows, cols;
    ifs >> rows >> cols;
    mat.resize(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ifs >> mat(i, j);
        }
    }

    ifs.close();

    return 0;
}

template <typename SparseMat>
int write_sparse_mat(const std::string &out_file,
                     const SparseMat &spmat) {
    using namespace std;
    ofstream ofs(out_file);
    if (!ofs) {
        cerr << "matrix file creat error: " << out_file << endl;
        return __LINE__;
    }

    for (int k = 0; k < spmat.outerSize(); ++k)
        for (typename SparseMat::InnerIterator it(spmat, k); it; ++it) {
            ofs << it.row() << ", " << it.col() << ", " << it.value() << endl;
        }

    ofs.close();
    return 0;
}

template <typename SparseMat>
int read_sparse_mat(const std::string &in_file,
                    SparseMat &spmat) {
    return 0;
    // TODO: implement
    std::cerr << __FUNCTION__ << " not implemented " << std::endl;
}

} // namespace zxh::file_io