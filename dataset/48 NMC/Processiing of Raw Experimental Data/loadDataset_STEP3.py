import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
def load_week_Dataset(weekNum, Q_discharge_2A, V_discharge_2A, t_discharge_2A, I_discharge_2A, E_discharge_2A):
    if weekNum == 0:
        path = "C:/Users/jinqiang/Dropbox/IEC Project Materials/Code/RWTH-2021-04545_818642/characterization/BOL Part 1/"
    if weekNum >= 1:
        path = f"C:/Users/jinqiang/Dropbox/IEC Project Materials/Code/RWTH-2021-04545_818642/characterization/Characterization after {str(weekNum)} weeks (6.5 days) of cycling/"
    files = glob.glob(path + '*.csv')
    for k, file in enumerate(files):
        terms = file.split("/")
        last_term = terms[-1]
        last_folder, file_name = last_term.split("\\")
        cell = file_name[12:15]
        print(cell)
        cell_file = pd.read_csv(file)
        cell_file = cell_file[['Zeit', 'Prozedur', 'AhStep', 'WhStep', 'Spannung', 'Strom']]
        cell_file = cell_file.dropna()
        cell_file_2A_dis = cell_file[cell_file['Prozedur'] == "TBA_SD"]
        Q_discharge_2A[cell, weekNum] = cell_file_2A_dis['AhStep'].to_numpy(dtype="float32")
        V_discharge_2A[cell, weekNum] = cell_file_2A_dis['Spannung'].to_numpy(dtype="float32")
        # t_discharge_2A[cell, weekNum] = cell_file_2A_dis['Zeit'].to_numpy(dtype="float32")
        I_discharge_2A[cell, weekNum] = cell_file_2A_dis['Strom'].to_numpy(dtype="float32")
        E_discharge_2A[cell, weekNum] = cell_file_2A_dis['WhStep'].to_numpy(dtype="float32")

        # select elements after the peak voltage
        index_max_voltage = np.argmax(V_discharge_2A[cell, weekNum])
        Q_discharge_2A[cell, weekNum] = Q_discharge_2A[cell, weekNum][index_max_voltage::]
        V_discharge_2A[cell, weekNum] = V_discharge_2A[cell, weekNum][index_max_voltage::]
        I_discharge_2A[cell, weekNum] = I_discharge_2A[cell, weekNum][index_max_voltage::]
        E_discharge_2A[cell, weekNum] = E_discharge_2A[cell, weekNum][index_max_voltage::]

        # select element before the discharge cutoff voltage
        index_cutoff_voltage = np.where(V_discharge_2A[cell, weekNum] < 3.0 + 4e-4)
        index_cutoff_voltage = index_cutoff_voltage[0][0]

        # index_min_voltage = np.argmin(V_discharge_2A[cell, weekNum])
        Q_discharge_2A[cell, weekNum] = Q_discharge_2A[cell, weekNum][0:index_cutoff_voltage]
        V_discharge_2A[cell, weekNum] = V_discharge_2A[cell, weekNum][0:index_cutoff_voltage]
        I_discharge_2A[cell, weekNum] = I_discharge_2A[cell, weekNum][0:index_cutoff_voltage]
        E_discharge_2A[cell, weekNum] = E_discharge_2A[cell, weekNum][0:index_cutoff_voltage]

        # print("here")
    return Q_discharge_2A, V_discharge_2A, I_discharge_2A

if __name__ == '__main__':
    Q_discharge_2A, V_discharge_2A, t_discharge_2A, I_discharge_2A, E_discharge_2A = {}, {}, {}, {}, {}
    # starting point
    start_week = 0
    load_week_Dataset(start_week, Q_discharge_2A, V_discharge_2A, t_discharge_2A, I_discharge_2A, E_discharge_2A)
    # end point
    end_week = 4
    load_week_Dataset(end_week, Q_discharge_2A, V_discharge_2A, t_discharge_2A, I_discharge_2A, E_discharge_2A)

    cells = ["00" + str(i) for i in range(2, 10)] + ["0" + str(i) for i in range(10, 50)]
    fig_width, fig_height = 4, 4
    nrows, ncols = 10, 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width * ncols, fig_height * nrows))
    for row in range(nrows):
        for col in range(ncols):
            if row * ncols + col > 47:
                break
            cell = cells[row * ncols + col]
            ax[row][col].plot(Q_discharge_2A[cell, start_week], V_discharge_2A[cell, start_week], linestyle='-', label='BOL')
            print(f"Q_discharge_2A of the cell {cell} at BOL", Q_discharge_2A[cell, start_week].shape)
            ax[row][col].plot(Q_discharge_2A[cell, end_week], V_discharge_2A[cell, end_week], linestyle='-',
                              label=f'Characterization after the {end_week}th weeks of cycling test')
            print(f"Q_discharge_2A of the cell {cell} for the characterization test after the {end_week} weeks of cycling test", Q_discharge_2A[cell, end_week].shape)
    ax[0][0].legend()
    plt.show()
