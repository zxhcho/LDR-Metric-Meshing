import numpy as np

# read vtk version 3.0 with vertex property


def read_vtk(file_name):
    with open(file_name, 'r') as f:
        while True:
            line = f.readline().strip().split()
            if len(line) == 0:
                continue
            if line[0] == "POINTS":
                point_num = int(line[1])
                V = np.zeros([point_num, 3])
                for i in range(point_num):
                    line = f.readline().strip().split()
                    for j in range(3):
                        V[i, j] = float(line[j])
            elif line[0] == "CELLS":
                cell_num = int(line[1])
                total_num = int(line[2])
                vertices_per_cell = int(total_num / cell_num - 1)
                C = np.zeros([cell_num, vertices_per_cell], dtype=int)
                for ci in range(cell_num):
                    line = f.readline().strip().split()
                    while (len(line) == 0):
                        line = f.readline().strip().split()
                    assert (int(line[0]) == vertices_per_cell)
                    for j in range(vertices_per_cell):
                        C[ci, j] = int(line[j+1])
                break
    return V, C


def read_vtk_with_attribute(file_name):
    with open(file_name, 'r') as f:
        while True:
            line = f.readline().strip().split()
            if len(line) == 0:
                continue
            if line[0] == "POINTS":
                point_num = int(line[1])
                V = np.zeros([point_num, 3])
                for i in range(point_num):
                    line = f.readline().strip().split()
                    for j in range(3):
                        V[i, j] = float(line[j])
            elif line[0] == "CELLS":
                cell_num = int(line[1])
                total_num = int(line[2])
                vertices_per_cell = int(total_num / cell_num - 1)
                C = np.zeros([cell_num, vertices_per_cell], dtype=int)
                for ci in range(cell_num):
                    line = f.readline().strip().split()
                    while (len(line) == 0):
                        line = f.readline().strip().split()
                    assert (int(line[0]) == vertices_per_cell)
                    for j in range(vertices_per_cell):
                        C[ci, j] = int(line[j+1])
            elif line[0] == "CELL_DATA" or line[0] == "POINT_DATA":
                n = int(line[1])
                # print("n field = ", n)
                P = np.zeros(n)
                for i in range(2):
                    f.readline()
                i = 0
                while (i < n):
                    line = f.readline().strip().split()
                    if (len(line) == 0):
                        continue
                    P[i] = float(line[0])
                    i += 1
                break
    return V, C, P


def read_vtk_attribute(file_name):
    """only read property
    """
    with open(file_name, 'r') as f:
        while True:
            line = f.readline().strip().split()
            if len(line) == 0:
                continue
            if line[0] == "POINTS":
                point_num = int(line[1])
                for _ in range(point_num):
                    line = f.readline().strip().split()
                    while (len(line) == 0):
                        line = f.readline().strip().split()
            elif line[0] == "CELLS":
                cell_num = int(line[1])
                for _ in range(cell_num):
                    line = f.readline().strip().split()
                    while (len(line) == 0):
                        line = f.readline().strip().split()
            elif line[0] == "CELL_DATA" or line[0] == "POINT_DATA":
                n = int(line[1])
                P = np.zeros(n)
                for i in range(2):
                    f.readline()
                i = 0
                while (i < n):
                    line = f.readline().strip().split()
                    if (len(line) == 0):
                        continue
                    P[i] = float(line[0])
                    i += 1
                break
    return P


def read_vtk_with_attributes(file_name):
    with open(file_name, 'r') as f:
        P = []
        while True:
            line = f.readline().strip().split()
            if len(line) == 0:
                break
            if line[0] == "POINTS":
                point_num = int(line[1])
                V = np.zeros([point_num, 3])
                for i in range(point_num):
                    line = f.readline().strip().split()
                    for j in range(3):
                        V[i, j] = float(line[j])
            elif line[0] == "CELLS":
                cell_num = int(line[1])
                total_num = int(line[2])
                vertices_per_cell = int(total_num / cell_num - 1)
                C = np.zeros([cell_num, vertices_per_cell], dtype=int)
                for ci in range(cell_num):
                    line = f.readline().strip().split()
                    assert (int(line[0]) == vertices_per_cell)
                    for j in range(vertices_per_cell):
                        C[ci, j] = int(line[j+1])
            elif line[0] == "POINT_DATA":
                n = int(line[1])
                print("n field = ", n)
                while True:
                    prop = np.zeros(n)
                    line = f.readline().strip().split()
                    if len(line) == 0:
                        break
                    if line[0] == "SCALARS":
                        f.readline()
                        for i in range(n):
                            line = f.readline().strip().split()
                            prop[i] = float(line[0])
                        P.append(prop)
                    elif line[0] == "CELL_DATA":
                        n = int(line[1])
                        print("n field = ", n)
                        while True:
                            prop = np.zeros(n)
                            for i in range(2):
                                f.readline()
                            for i in range(n):
                                line = f.readline().strip().split()
                                prop[i] = float(line[0])
                            P.append(prop)
            elif line[0] == "CELL_DATA":
                n = int(line[1])
                print("n field = ", n)
                while True:
                    prop = np.zeros(n)
                    line = f.readline().strip().split()
                    line = f.readline().strip().split()
                    if (len(line) == 0):
                        break
                    for i in range(n):
                        line = f.readline().strip().split()
                        if (len(line) == 0):
                            break
                        prop[i] = float(line[0])
                    P.append(prop)
                break

    return V, C, P


def write_vtk(file_name, V, C):
    surface_cell_t = {
        1: 1,
        2: 3,
        3: 5,
        4: 9
    }

    with open(file_name, 'w') as ofs:
        num_points = V.shape[0]
        num_cells = C.shape[0]
        point_list_size = 0
        for cell in C:
            point_list_size += len(cell) + 1

        # file header
        ofs.write("# vtk DataFile Version 3.0\n")
        ofs.write("Mesh Generated by file_io\n")
        ofs.write("ASCII\n")
        ofs.write("DATASET UNSTRUCTURED_GRID\n")
        ofs.write("POINTS " + str(num_points) + " double\n")

        # point data
        for i in range(num_points):
            ofs.write(str(V[i, 0]) + ' ' + str(V[i, 1]) +
                      ' ' + str(V[i, 2]) + '\n')

        # cell structure
        ofs.write("CELLS " + str(num_cells) +
                  ' ' + str(point_list_size) + '\n')
        for cell in C:
            ofs.write(str(len(cell)) + ' ')
            for v in cell:
                ofs.write(str(v) + ' ')
            ofs.write('\n')

        # cell types
        ofs.write("CELL_TYPES " + str(num_cells) + '\n')
        for cell in C:
            ofs.write(str(surface_cell_t[len(cell)]) + '\n')


def write_vtk_with_attribute(file_name, V, C, P, name="Property"):
    surface_cell_t = {
        1: 1,
        2: 3,
        3: 5,
        4: 9
    }

    with open(file_name, 'w') as ofs:
        num_points = V.shape[0]
        num_cells = C.shape[0]
        point_list_size = 0
        for cell in C:
            point_list_size += len(cell) + 1

        # file header
        ofs.write("# vtk DataFile Version 3.0\n")
        ofs.write("Mesh Generated by file_io\n")
        ofs.write("ASCII\n")
        ofs.write("DATASET UNSTRUCTURED_GRID\n")
        ofs.write("POINTS " + str(num_points) + " double\n")

        # point data
        for i in range(num_points):
            ofs.write(str(V[i, 0]) + ' ' + str(V[i, 1]) +
                      ' ' + str(V[i, 2]) + '\n')

        # cell structure
        ofs.write("CELLS " + str(num_cells) +
                  ' ' + str(point_list_size) + '\n')
        for cell in C:
            ofs.write(str(len(cell)) + ' ')
            for v in cell:
                ofs.write(str(v) + ' ')
            ofs.write('\n')

        # cell types
        ofs.write("CELL_TYPES " + str(num_cells) + '\n')
        for cell in C:
            ofs.write(str(surface_cell_t[len(cell)]) + '\n')

        data_type = None
        data_dimension = None
        # * judge cell or point data
        if P.shape[0] == num_points:
            data_type = "POINT_DATA " + str(P.shape[0]) + '\n'
        elif P.shape[0] == num_cells:
            data_type = "CELL_DATA " + str(P.shape[0]) + '\n'
        else:
            print("P number error")

        if len(P.shape) == 1 or P.shape[1] == 1:
            # * scalar
            data_dimension = "SCALARS " + name + " double 1\n LOOKUP_TABLE default \n"
        elif P.shape[1] > 1:
            data_dimension = "VECTORS " + name + " double \n"
        ofs.write(data_type)
        ofs.write(data_dimension)
        for row in P:
            if isinstance(row, (int, float, np.float32, np.float64, np.int32, np.int64)):
                ofs.write(str(row))
            else:
                for col in row:
                    ofs.write(str(col) + ' ')
            ofs.write('\n')
