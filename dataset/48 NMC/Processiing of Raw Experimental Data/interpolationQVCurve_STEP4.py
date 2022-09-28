import numpy as np
from load_dataset_STEP3 import  load_week_Dataset
from scipy import interpolate
from matplotlib import pyplot as plt

def interpolation(x, y, x_new, smoothness):
    # sort the input in ascending order
    p = x.argsort()
    x = x[p]
    y = y[p]
    tck = interpolate.splrep(x, y, s=smoothness)
    y_new = interpolate.splev(x_new, tck, der=0)
    dydx_new = interpolate.splev(x_new, tck, der=1)
    # plt.plot(x, y)
    # plt.plot(x_new, y_new)
    # plt.plot(x_new, dydx_new)
    # plt.show()

    return y_new, dydx_new
if __name__ == "__main__":
    path = "C:/Users/jinqiang/Dropbox/IEC Project Materials/Code/RWTH-2021-04545_818642/characterization/Interpolation results/"
    # Load datasets for the BOL test and the characterization test after the first cycling test
    Q_discharge_2A, V_discharge_2A, t_discharge_2A, I_discharge_2A, E_discharge_2A = {}, {}, {}, {}, {}

    for week in range(0, 19):
        print(week)
        load_week_Dataset(week, Q_discharge_2A, V_discharge_2A, t_discharge_2A, I_discharge_2A, E_discharge_2A)

    # interpolation and save results
    cells = ["00" + str(i) for i in range(2, 10)] + ["0" + str(i) for i in range(10, 50)]
    voltage_new = np.linspace(3.0, 4.0, 1000)
    for cell in cells:
        QV, dQdV = np.array([]).reshape(voltage_new.shape[0], -1), np.array([]).reshape(voltage_new.shape[0], -1)
        for week in range(19):
            if (cell, week)  not in V_discharge_2A:
                break
            QV_new_week, dQdV_new_week = interpolation(V_discharge_2A[cell, week], Q_discharge_2A[cell, week], voltage_new, 8e-4)
            QV = np.column_stack((QV, QV_new_week))
            dQdV = np.column_stack((dQdV, dQdV_new_week))
            # QV[:, 1], dQdV[:, 1] = interpolation(V_discharge_2A[cell, 1], Q_discharge_2A[cell, 1], voltage_new, 8e-4)
        np.savetxt(path + "QV data/" + cell + ".csv", QV, delimiter=",")
        np.savetxt(path + "dQdV data/" + cell + ".csv", dQdV, delimiter=",")