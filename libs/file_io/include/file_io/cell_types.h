#pragma once

#include <map>

namespace zxh::file_io {

enum class vtk_file_t : int {
    surface = 0,
    volume = 1,
};

enum class vtk_cell_t : int {
    vertex = 1,
    poly_vertex = 2,
    line = 3,
    poly_line = 4,
    tri = 5,
    tri_strip = 6,
    polygon = 7,
    pixel = 8,
    quad = 9,
    tet = 10,
    voxel = 11,
    hex = 12,
    wedge = 13,
    pyramid = 14,
    quadric_hex = 25
};

//* map from n_points in a cell to its cell type
const static std::map<int, int> surface_cell_t{
    {1, (int)vtk_cell_t::vertex},
    {2, (int)vtk_cell_t::line},
    {3, (int)vtk_cell_t::tri},
    {4, (int)vtk_cell_t::quad}};
const static std::map<int, int> volumn_cell_t{
    {4, (int)vtk_cell_t::tet},
    {5, (int)vtk_cell_t::pyramid},
    {6, (int)vtk_cell_t::wedge},
    {8, (int)vtk_cell_t::hex},
    {20, (int)vtk_cell_t::quadric_hex}};

} // namespace zxh::file_io