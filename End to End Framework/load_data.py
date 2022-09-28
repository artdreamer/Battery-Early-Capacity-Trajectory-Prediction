"""Load capacity trajectories and QV curves from csv files."""

import numpy as np
def load_dataset(work_path, folder, valid_cells, dataype="capacity trajectory"):
    """
    Load data from .csv files.

    :param work_path:
    :param folder:
    :param valid_cells:
    :param dataype:
    :return:
    """
    if dataype == "capacity trajectory":
        data_curve = []
        csv_format = 1
    elif dataype == "voltage curve":
        num_cells = len(valid_cells)
        csv_format = 0
        data_curve = np.zeros((num_cells, 1000, 150), dtype="float32")
    prefix = ""

    if folder == "Capacity trajectories":
        prefix = "epSanyo"

    for k, cell_name in enumerate(valid_cells):
        file = work_path + f'/{folder}/' + prefix + cell_name + '.csv'
        cell = np.genfromtxt(file, delimiter=',', dtype="float32")
        if csv_format == 0:
            data_curve[k,:,0:cell.shape[1]] = cell
        else:
            data_curve.append(cell)
    return data_curve
