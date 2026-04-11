# -*-coding:utf-8-*-
import os
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
from scipy.signal import argrelextrema
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

__all__ = ['find_period',
           'find_length_rank',
           'load_dataset',
           "SequenceWindowConversion",
           "get_dataloader",
           "anomaly_score_func",
           "Multiple_dataset",
           "get_multiple_dataloader"]

def find_period(data: np.array):
    if len(data.shape) > 1:
        return 0
    data = data[:min(200000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 20 or local_max[max_local_max] > 1000:
            freq, power = periodogram(data)
            period = int(1 / freq[np.argmax(power)])
        else:
            period = local_max[max_local_max] + base
    except:
        freq, power = periodogram(data)
        period = int(1 / (freq[np.argmax(power)] + 1e-4))

    return period


def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape) > 1: return 0
    if rank == 0: return 1
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]  # Ascending order
        max_local_max = sorted_local_max[0]  # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    max_local_max = i
                    break
        if rank == 3:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]:
                    max_local_max = i
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except:
        return 125
    
def load_dataset(data_name: str, group_name: str = "1"):
    """
    Load dataset from the given data_name and group_name.
    :param data_name: UCR
    :param group_name: 1, 2, ..., 250
    :return: train_data, test_data, test_label, subsequence_list
    """
    if data_name == 'UCR':
        group_name = [int(i) - 1 for i in group_name]

        data_path = "dataset/UCR/processed"
        train_data_files = os.listdir(data_path + "/train")
        train_data_files = sorted(train_data_files)
        train_data_list = [train_data_files[i] for i in group_name]

        test_data_files = os.listdir(data_path + "/test")
        test_data_files = sorted(test_data_files)
        test_data_list = [test_data_files[i] for i in group_name]

        label_files = os.listdir(data_path + "/label")
        label_files = sorted(label_files)
        test_label_list = [label_files[i] for i in group_name]

        train_data = [np.load(os.path.join(data_path, "train", file)) for file in train_data_list]
        test_data = [np.load(os.path.join(data_path, "test", file)) for file in test_data_list]
        test_label = [np.load(os.path.join(data_path, "label", file)) for file in test_label_list]
        subsequence_list = pd.read_csv("dataset/UCR/all_length.csv")
        subsequence_list.sort_values(by="file_name", inplace=True)
        subsequence_list = subsequence_list['period'].tolist()
        subsequence_list = [subsequence_list[i] for i in group_name]
        return train_data, test_data, test_label, subsequence_list
    elif data_name == "TSB-AD":
        group_name = [int(i) - 1 for i in group_name]
        data_path = "dataset/TSB-AD/raw"
        data_files = os.listdir(data_path)
        data_files = sorted(data_files)
        data_list = [data_files[i] for i in group_name]
        train_data_list = []
        test_data_list = []
        test_label = []
        subsequence_list = []
        for file in data_list:
            df = pd.read_csv(os.path.join(data_path, file)).dropna()
            test_label.append(df['Label'].astype(int).to_numpy())

            train_index = file.split('.')[0].split('_')[-3]
            train_data = df.iloc[:int(train_index), 0:-1].values.astype(float).squeeze()
            min_value, max_value = train_data.min(), train_data.max()
            train_data = (train_data - min_value) / (max_value - min_value)
            train_data_list.append(train_data)

            test_data = df.iloc[:, 0:-1].values.astype(float).squeeze()
            test_data = (test_data - min_value) / (max_value - min_value)
            test_data_list.append(test_data)
            subsequence_length_0 = find_length_rank(df.iloc[:, 0], rank=1)
            subsequence_length_1 = find_period(df.iloc[:, 0])
            subsequence_length = min(subsequence_length_0, subsequence_length_1)
            subsequence_list.append(subsequence_length)
        return train_data_list, test_data_list, test_label, subsequence_list
    elif data_name == "ASD":
        group_name = [int(i) - 1 for i in group_name]
        data_path = "dataset/ASD/processed"
        data_files = os.listdir(data_path)
        train_files = [file for file in data_files if 'train' in file]
        test_files = [file for file in data_files if 'test' in file and 'label' not in file]
        label_files = [file for file in data_files if 'label' in file]
        train_files = sorted(train_files)
        test_files = sorted(test_files)
        label_files = sorted(label_files)

        train_data_list = []
        test_data_list = []
        test_label = []
        subsequence_list = []

        for i in group_name:
            with open(os.path.join(data_path, train_files[i]), 'rb') as f:
                train_data_list.append(pickle.load(f))

            with open(os.path.join(data_path, test_files[i]), 'rb') as f:
                test_data_list.append(pickle.load(f))

            with open(os.path.join(data_path, label_files[i]), 'rb') as f:
                test_label.append(pickle.load(f))

            subsequence_list.append(find_length_rank(train_data_list[-1][:, -1], rank=1))

        return train_data_list, test_data_list, test_label, subsequence_list

    else:
        raise ValueError("Invalid data_name!")


class SequenceWindowConversion:
    def __init__(self, window_size: int, stride_size: int = 1):
        """
        @param window_size: window size
        @param stride_size: moving size
        """

        self.windows = None
        self.pad_sequence_data = None
        self.raw_sequence_data = None
        self.pad_length = None
        self.is_converted = False
        self.window_size = window_size
        self.stride_size = stride_size
        assert stride_size <= window_size, "window size must be larger than stride size"

    def sequence_to_windows(self, sequence_data) -> np.ndarray:
        """
        @param sequence_data: (length, channels)
        @return: windows: (num_window, window_size, channels)
        """
        self.is_converted = True
        self.raw_sequence_data = sequence_data
        raw_data_length, num_channels = self.raw_sequence_data.shape
        # pad the first patch
        pad_length: int = self.stride_size - (raw_data_length - self.window_size) % self.stride_size
        if self.stride_size == 1 or (raw_data_length - self.window_size) % self.stride_size == 0:
            pad_length = 0
        self.pad_length = pad_length
        self.pad_sequence_data = np.concatenate([np.zeros([pad_length, num_channels]), sequence_data], axis=0)
        data_length, num_channels = self.pad_sequence_data.shape

        start_idx_list = np.arange(0, data_length - self.window_size + 1, self.stride_size)
        end_idx_list = np.arange(self.window_size, data_length + 1, self.stride_size)
        windows = []

        for start_id, end_id in zip(start_idx_list, end_idx_list):
            windows.append(self.pad_sequence_data[start_id:end_id])

        self.windows = np.array(windows, dtype=np.float32)

        return self.windows

    def windows_to_sequence(self, windows: np.ndarray) -> np.ndarray:
        """
        convert the windows back to same length sequence, where the overlapping parts take the mean value
        @param windows: (num_window, window_size, channels)
        @return: sequence_data: (length, channels)
        """
        assert self.is_converted, "please first convert to windows"
        # initialize an empty array to store the sequence data
        sequence_data = np.zeros_like(self.pad_sequence_data)
        # get the number of windows, the window size, and the number of channels
        num_window, window_size, num_channels = windows.shape
        # get the length of the original sequence data
        length = sequence_data.shape[0]
        # loop through each window
        for i in range(num_window):
            # get the start and end index of the window in the sequence data
            start = i * self.stride_size
            end = start + window_size
            # if the end index exceeds the length, truncate the window
            if end > length:
                end = length
                window = windows[i, :end - start, :]
            else:
                window = windows[i]
            # add the window to the corresponding part of the sequence data
            sequence_data[start:end, :] += window
        # get the number of times each element in the sequence data is added
        counts = np.zeros_like(sequence_data)
        # loop through each window again
        for i in range(num_window):
            # get the start and end index of the window in the sequence data
            start = i * self.stride_size
            end = start + window_size
            # if the end index exceeds the length, truncate the window
            if end > length:
                end = length
            # increment the counts by one for each element in the window
            counts[start:end, :] += 1
        # divide the sequence data by the counts to get the mean value
        sequence_data /= counts
        # return the sequence data
        return sequence_data[self.pad_length:, :]


def get_dataloader(data: np.ndarray, batch_size: int, window_length: int, train_stride: int = None,
                   test_stride: int = None, mode="test"):
    assert mode in ["train", "test"]
    if mode == "train":
        if_shuffle = True
        if train_stride is None:
            stride_size = 4
        else:
            stride_size = train_stride
    else:
        stride_size = test_stride
        batch_size = 2 * batch_size
        if_shuffle = False

    window_converter = SequenceWindowConversion(
        window_size=window_length,
        stride_size=stride_size
    )
    windows = window_converter.sequence_to_windows(data)
    windows = torch.tensor(windows)
    dataset = TensorDataset(windows)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=if_shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4
    )

    return data_loader, window_converter


def anomaly_score_func(raw_value: np.ndarray, predict_value: np.ndarray, subsequence: int):
    """
    calculate the anomaly score for a given subsequence
    :param raw_value:
    :param predict_value:
    :param subsequence: moveing average window size
    :return: anomaly_score
    """
    if len(raw_value.shape) > 1:
        raw_value = np.squeeze(raw_value)
    if len(predict_value.shape) > 1:
        predict_value = np.squeeze(predict_value)
    anomaly_score = np.abs(raw_value - predict_value)
    weight = np.ones(subsequence) / subsequence
    anomaly_score = np.convolve(anomaly_score, weight, mode='same')
    return anomaly_score


class Multiple_dataset(Dataset):
    def __init__(self, train_window_list):
        self.train_window_list = train_window_list

    def __len__(self):
        lengths = [len(train_window) for train_window in self.train_window_list]
        return max(lengths)

    def __getitem__(self, index):
        return_windows = []
        for train_window in self.train_window_list:
            if index < len(train_window):
                return_windows.append(train_window[index])
            else:
                random_index = np.random.randint(0, len(train_window))
                return_windows.append(train_window[random_index])
        return return_windows


def get_multiple_dataloader(train_data_list,
                            batch_size,
                            window_length_list,
                            train_stride):
    window_converter_list = [
        SequenceWindowConversion(
            window_size=window_length,
            stride_size=train_stride
        ) for window_length in window_length_list]

    train_windows_list = [
        window_converter.sequence_to_windows(train_data)
        for window_converter, train_data in zip(window_converter_list, train_data_list)
    ]

    train_dataset = Multiple_dataset(train_windows_list)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    return train_data_loader


if __name__ == '__main__':
    train_data, test_data, test_label, subsequence_list = load_dataset(
        data_name="ASD",
        group_name=["10"]
    )
    print(len(train_data))
