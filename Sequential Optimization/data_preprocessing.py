"""Data preprocessing modules."""
import numpy as np
from scipy import interpolate
from scipy.stats import skew, kurtosis
from generate_dataset import capacity_trajectory_train_test_split
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data import DataLoader
from common import device_info
import torch
from load_data import load_dataset


class ExpertDataSet(Dataset):
    """Orgnize the dataset."""
    def __init__(
        self, input_dataset, fitted_params, capacity_trajectories, cycle_lives, initial_capacities
    ):
        self.input = input_dataset
        self.fitted_params = fitted_params
        self.time = capacity_trajectories[:, :, 0]
        self.capacity = capacity_trajectories[:, :, 1]
        self.diff_capacity = capacity_trajectories[:, :, 2]
        self.cycle_lives = cycle_lives
        self.initial_caps = initial_capacities

    def __getitem__(self, index):
        return (
            self.input[index], self.fitted_params[index], self.time[index, :],
            self.capacity[index, :], self.diff_capacity[index, :],
            self.cycle_lives[index], self.initial_caps[index]
        )

    def __len__(self):
        return len(self.capacity)


def get_early_featuer(delta_QVCurve, capacity_trajectories, initial_capacity, precision):
    """Extract the six early-life fetures."""
    max_capacity = np.full((len(capacity_trajectories)), np.nan, dtype=precision)
    for idx, capacity_trajectory in enumerate(capacity_trajectories):
        max_capacity[idx] = capacity_trajectory[:,1].max()
    x_1 = initial_capacity
    x_2 =  max_capacity - initial_capacity
    x_3 = np.log10(np.abs(np.min(delta_QVCurve, axis=1)))
    x_4 = np.log10(np.abs(np.var(delta_QVCurve, axis=1)))
    x_5 = np.log10(np.abs(skew(delta_QVCurve, axis=1)))
    x_6 = np.log10(np.abs(kurtosis(delta_QVCurve, axis=1)))
    return np.column_stack((x_1, x_2, x_3, x_4, x_5, x_6))

def normalize_capacity_trajectory(capacity_trjectories):
    inital_caps = np.full((len(capacity_trjectories),), np.nan)
    for idx, _ in enumerate(capacity_trjectories):
        inital_caps[idx] = capacity_trjectories[idx][0, 2]
        capacity_trjectories[idx][:, 1] = capacity_trjectories[idx][:, 1] / inital_caps[idx]
    return inital_caps


def sample_trajectory_no_interpolation(trajectory, num_points_trajectory=100):
    num_cycle = len(trajectory)
    sample_cycles = np.linspace(0, num_cycle - 1, num_points_trajectory)
    return trajectory[sample_cycles.astype(int), :]


def sample_trajectory_with_interpolation(trajectory, num_points_trajectory=100):
    cycles = trajectory[:, 0]
    capacities = trajectory[:, 1]
    tck = interpolate.splrep(cycles, capacities, s=0)
    new_cycles = np.linspace(2, cycles[-1], 100)
    sample_trajectory = np.full((num_points_trajectory, 3), np.nan)
    sample_trajectory[:, 0] = new_cycles
    sample_trajectory[:, 1] = interpolate.splev(new_cycles, tck, der=0)
    sample_trajectory[:, 2] = interpolate.splev(new_cycles, tck, der=1)
    return sample_trajectory


def prepare_data_train(
        dataset_name, input_type, raw_data_path, precision, model_name, params, batch_size=16
    ):
    if dataset_name == "48 NMC":
        # load valid cells, initial capacities, and cycle lives
        # raw_data_path = '../dataset/48 NMC'
        valid_cells = ["00" + str(i) for i in range(2, 10)] + ["0" + str(i) for i in range(10, 50)]
        initial_capacity = np.genfromtxt(
            raw_data_path + '/initial_capacities.csv', delimiter=',', dtype=precision
        )
        cycle_lives = np.genfromtxt(
            raw_data_path + '/cycle_lives/cyclelives_80percent.csv', delimiter=',',
            dtype=precision
        )

        # load target output
        capacity_trajectories = load_dataset(raw_data_path, "Capacity trajectories", valid_cells)

        # load raw inputs
        QVCurve = load_dataset(
            raw_data_path, folder="QV data", valid_cells=valid_cells, dataype="voltage curve"
        )
        dQdV = load_dataset(
            raw_data_path, folder="dQdV data", valid_cells=valid_cells, dataype="voltage curve"
        )

        # Construct inputs: difference of QV curves between week one and week zeros
        DeltaQ_w1_minus_w0 = QVCurve[:, :, 1] - QVCurve[:, :, 0]
        Delta_dQ_dV_w1_minus_w0 = dQdV[:, :, 1] - dQdV[:, :, 0]

        if input_type == "delta QV 100-10":
            input_data = DeltaQ_w1_minus_w0[:, ::10]
        elif input_type == "delta QV and dQdV 100-10":
            input_data =  np.concatenate(
                (DeltaQ_w1_minus_w0, Delta_dQ_dV_w1_minus_w0), axis=1
            )[:, ::10]
        elif input_type == "discharge features":
            input_data = get_early_featuer(
                DeltaQ_w1_minus_w0, capacity_trajectories, initial_capacity, precision
            )
        else:
            raise ValueError(f"{input_type} is not a valid input type")

        # input normalization
        input_data = (input_data - input_data.mean()) / input_data.std()

        # split the data into train and test datasets
        capacity_trajectories_train, capacity_trajectories_test, idx_train, \
            idx_test = capacity_trajectory_train_test_split(capacity_trajectories, split=0.8)

        # Construct output
        fitted_params_train = np.genfromtxt(raw_data_path + '/fitted empirical models/' + model_name
                                            + ' parameters.csv', delimiter=',', dtype="float32")
        fitted_params_test = np.random.rand(len(idx_test), len(params)).astype(dtype="float32")
        num_points_trajectory = 100
        formatted_capacity_trajectories_train = np.full(
            (len(idx_train), num_points_trajectory, 3), np.nan
        )
        formatted_capacity_trajectories_test = np.full(
            (len(idx_test), num_points_trajectory, 3), np.nan
        )
        for idx, trajectory in enumerate(capacity_trajectories_train):
            formatted_capacity_trajectories_train[idx, :, :] = sample_trajectory_no_interpolation(
                trajectory, num_points_trajectory=num_points_trajectory
            )
        for idx, trajectory in enumerate(capacity_trajectories_test):
            formatted_capacity_trajectories_test[idx, :, :] = sample_trajectory_no_interpolation(
                trajectory, num_points_trajectory=num_points_trajectory
            )

        train_dataset = ExpertDataSet(
            input_data[idx_train], fitted_params_train, formatted_capacity_trajectories_train,
            cycle_lives[idx_train], initial_capacity[idx_train]
        )
        test_dataset = ExpertDataSet(
            input_data[idx_test], fitted_params_test, formatted_capacity_trajectories_test,
            cycle_lives[idx_test], initial_capacity[idx_test]
        )


        training_size = int(0.7 * len(train_dataset))
        val_size = len(train_dataset) - training_size
        training_dataset, val_dataset = random_split(
            train_dataset, [training_size, val_size], generator=torch.Generator().manual_seed(7)
        )

        # data format, device info, and data loader
        device, kwargs = device_info()

        train_dataloader = DataLoader(
            dataset=training_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=1, shuffle=True, **kwargs
        )
        return train_dataloader, val_dataloader, device, train_dataset, [test_dataset]

    elif dataset_name == "169 LFP":
        # load valid cells, initial capacities, and cycle lives
        # raw_data_path = '../dataset/169 LFP'

        # load cycle lives for the train and three test datasets
        cycle_lives_train = np.genfromtxt(raw_data_path + '/cycle_lives/train_cycle_lives.csv',
                                          delimiter=',', dtype=precision)
        cycle_lives_test1 = np.genfromtxt(raw_data_path + '/cycle_lives/test1_cycle_lives.csv',
                                          delimiter=',', dtype=precision)
        cycle_lives_test2 = np.genfromtxt(raw_data_path + '/cycle_lives/test2_cycle_lives.csv',
                                          delimiter=',', dtype=precision)
        cycle_lives_test3 = np.genfromtxt(raw_data_path + '/cycle_lives/test3_cycle_lives.csv',
                                          delimiter=',', dtype=precision)

        # load # load target output for the train and three test datasets
        valid_cells_train = ["cell" + str(i) for i in range(1, 42)]
        valid_cells_test1 = ["cell" + str(i) for i in range(1, 43)]
        valid_cells_test2 = ["cell" + str(i) for i in range(1, 41)]
        valid_cells_test3 = ["cell" + str(i) for i in range(1, 46)]
        capacity_trajectories_train = load_dataset(raw_data_path, "discharge_capacity/train",
                                                  valid_cells_train)
        capacity_trajectories_test1 = load_dataset(raw_data_path, "discharge_capacity/test1",
                                                  valid_cells_test1)
        capacity_trajectories_test2 = load_dataset(
            raw_data_path, "discharge_capacity/test2", valid_cells_test2
        )
        capacity_trajectories_test3 = load_dataset(
            raw_data_path, "discharge_capacity/test3", valid_cells_test3
        )

        # Normalize the capacity trajectory
        inital_caps_train = normalize_capacity_trajectory(capacity_trajectories_train)
        inital_caps_test1 = normalize_capacity_trajectory(capacity_trajectories_test1)
        inital_caps_test2 = normalize_capacity_trajectory(capacity_trajectories_test2)
        inital_caps_test3 = normalize_capacity_trajectory(capacity_trajectories_test3)

        num_points_trajectory = 100
        formatted_capacity_trajectories_train = np.full(
            (len(valid_cells_train), num_points_trajectory, 3), np.nan
        )
        formatted_capacity_trajectories_test1 = np.full(
            (len(valid_cells_test1), num_points_trajectory, 3), np.nan
        )
        formatted_capacity_trajectories_test2 = np.full(
            (len(valid_cells_test2), num_points_trajectory, 3), np.nan
        )
        formatted_capacity_trajectories_test3 = np.full(
            (len(valid_cells_test3), num_points_trajectory, 3), np.nan
        )
        for idx, trajectory in enumerate(capacity_trajectories_train):
            formatted_capacity_trajectories_train[idx, :, :] = sample_trajectory_with_interpolation(
                trajectory, num_points_trajectory=num_points_trajectory)
        for idx, trajectory in enumerate(capacity_trajectories_test1):
            formatted_capacity_trajectories_test1[idx, :, :] = sample_trajectory_with_interpolation(
                trajectory, num_points_trajectory=num_points_trajectory)
        for idx, trajectory in enumerate(capacity_trajectories_test2):
            formatted_capacity_trajectories_test2[idx, :, :] = sample_trajectory_with_interpolation(
                trajectory, num_points_trajectory=num_points_trajectory)
        for idx, trajectory in enumerate(capacity_trajectories_test3):
            formatted_capacity_trajectories_test3[idx, :, :] = sample_trajectory_with_interpolation(
                trajectory, num_points_trajectory=num_points_trajectory)

        # load raw inputs
        QVCurveTrain = load_dataset(raw_data_path, folder="V_Q_curve/train",
                                   valid_cells=valid_cells_train, dataype="voltage curve")
        QVCurveTest1 = load_dataset(raw_data_path, folder="V_Q_curve/test1",
                                   valid_cells=valid_cells_test1, dataype="voltage curve")
        QVCurveTest2 = load_dataset(raw_data_path, folder="V_Q_curve/test2",
                                   valid_cells=valid_cells_test2, dataype="voltage curve")
        QVCurveTest3 = load_dataset(raw_data_path, folder="V_Q_curve/test3",
                                   valid_cells=valid_cells_test3, dataype="voltage curve")

        # Construct inputs: difference of QV curves between week one and week zeros
        DeltaQ_100_minus_10_train = QVCurveTrain[:, :, 98] - QVCurveTrain[:, :, 8]
        DeltaQ_100_minus_10_test1 = QVCurveTest1[:, :, 98] - QVCurveTest1[:, :, 8]
        DeltaQ_100_minus_10_test2 = QVCurveTest2[:, :, 98] - QVCurveTest2[:, :, 8]
        DeltaQ_100_minus_10_test3 = QVCurveTest3[:, :, 98] - QVCurveTest3[:, :, 8]

        # generate inputs
        if input_type == "delta QV 100-10":
            x_train = DeltaQ_100_minus_10_train[:,1::10].astype(precision)
            x_test1 = DeltaQ_100_minus_10_test1[:, 1::10].astype(precision)
            x_test2 = DeltaQ_100_minus_10_test2[:, 1::10].astype(precision)
            x_test3 = DeltaQ_100_minus_10_test3[:, 1::10].astype(precision)
        elif input_type == "discharge features":
            initial_capacities_train = np.array(
                [capacity_trajectory[0, 2] for capacity_trajectory in capacity_trajectories_train]
            )
            x_train = get_early_featuer(
                DeltaQ_100_minus_10_train, capacity_trajectories_train, initial_capacities_train,
                precision
            )
            initial_capacities_test1 = np.array(
                [capacity_trajectory[0, 2] for capacity_trajectory in capacity_trajectories_test1]
            )
            x_test1 = get_early_featuer(
                DeltaQ_100_minus_10_test1, capacity_trajectories_test1, initial_capacities_test1,
                precision
            )
            initial_capacities_test2 = np.array(
                [capacity_trajectory[0, 2] for capacity_trajectory in capacity_trajectories_test2]
            )
            x_test2 = get_early_featuer(
                DeltaQ_100_minus_10_test2, capacity_trajectories_test2,initial_capacities_test2,
                precision
            )
            initial_capacities_test3 = np.array(
                [capacity_trajectory[0, 2] for capacity_trajectory in capacity_trajectories_test3]
            )
            x_test3 = get_early_featuer(
                DeltaQ_100_minus_10_test3, capacity_trajectories_test3, initial_capacities_test3,
                precision
            )
        else:
            raise ValueError(f"{input_type} is not a valid input type")
        # Input normalization
        x_mean = np.mean(x_train, axis=0)
        x_std = np.std(x_train, axis=0)

        x_train_scale = (x_train - x_mean) / x_std
        x_test1_scale = (x_test1 - x_mean) / x_std
        x_test2_scale = (x_test2 - x_mean) / x_std
        x_test3_scale = (x_test3 - x_mean) / x_std

        # Construct output
        fitted_params_train = np.genfromtxt(
            raw_data_path + '/fitted empirical models/' + model_name \
            + ' parameters.csv', delimiter=',', dtype="float32"
        )
        fitted_params_test1 = np.random.rand(
            len(x_test1_scale), len(params)).astype(dtype="float32"
            )
        fitted_params_test2 = np.random.rand(
            len(x_test2_scale), len(params)).astype(dtype="float32"
        )
        fitted_params_test3 = np.random.rand(
            len(x_test3_scale), len(params)).astype(dtype="float32"
        )


        # build data loaders
        device, kwargs = device_info()
        train_dataset = ExpertDataSet(
            x_train_scale, fitted_params_train, formatted_capacity_trajectories_train,
            cycle_lives_train, inital_caps_train
        )
        test1_dataset = ExpertDataSet(
            x_test1_scale, fitted_params_test1, formatted_capacity_trajectories_test1,
            cycle_lives_test1, inital_caps_test1
        )
        test2_dataset = ExpertDataSet(
            x_test2_scale, fitted_params_test2, formatted_capacity_trajectories_test2,
            cycle_lives_test2, inital_caps_test2
        )
        test3_dataset = ExpertDataSet(
            x_test3_scale, fitted_params_test3, formatted_capacity_trajectories_test3,
            cycle_lives_test3, inital_caps_test3
        )

        # training and val split
        training_size = int(0.7 * len(train_dataset))
        val_size = len(train_dataset) - training_size
        training_dataset, eval_dataset = random_split(
            train_dataset, [training_size, val_size], generator=torch.Generator().manual_seed(7)
        )
        training_dataloader = DataLoader(
            dataset=training_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        val_dataloader = DataLoader(
            dataset=eval_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        return training_dataloader, val_dataloader, device, train_dataset, \
            [test1_dataset, test2_dataset, test3_dataset]

    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name." + \
                        "The current version only supports the datasets of" + \
                         "\"48 NMC\" and \"169 LFP\".")
