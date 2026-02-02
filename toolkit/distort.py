# -*-coding:utf-8-*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


__all__ = ['uniform_distort',
           'scale_distort',
           'jitering_distort',
           'sudden_add_drop',
           'frequency_distort',
           'point_distort',
           "original",
           "soft_raplacement",
           "mirror_flip"]

def uniform_distort(time_series, subsequence_length=None, inject_rate=None):
    """
    @param time_series: (batch_size, length, 1)
    @return:

    Parameters
    ----------
    subsequence_length
    :param inject_rate:
    :param subsequence_length:
    """
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    if subsequence_length is None:
        if inject_rate is None:
            rate = np.random.uniform(0.10, 0.15)
        else:
            rate = inject_rate
        replace_length = length
    else:
        if inject_rate is None:
            rate = np.random.uniform(0.1, 1)
        else:
            rate = inject_rate
        replace_length = subsequence_length

    replace_num = int(replace_length * rate)

    if replace_num == 0:
        replace_num += 2

    if replace_num >= length:
        replace_num = length - 2


    replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
    # select values to be replaced
    col_index = replace_index + torch.arange(replace_num)
    row_index = torch.arange(batch_size)[:, None]
    time_series[row_index, col_index, :] = time_series[row_index, replace_index, :]

    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1
    return time_series, label


def scale_distort(time_series, subsequence_length=None, inject_rate=None):
    """
    @param time_series: (batch_size, length, 1)
    @param rate: float, the rate of distortion
    @return:

    Parameters
    ----------
    subsequence_length
    """
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    if subsequence_length is None:
        if inject_rate is None:
            rate = np.random.uniform(0.10, 0.15)
        else:
            rate = inject_rate
        replace_length = length
    else:
        if inject_rate is None:
            rate = np.random.uniform(0.1, 1)
        else:
            rate = inject_rate
        replace_length = subsequence_length

    replace_num = int(replace_length * rate)

    if replace_num == 0:
        replace_num += 1

    if replace_num >= length:
        replace_num = length - 2

    replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
    scale = np.random.randint(2, 5)
    scaled_length = replace_num // scale
    replace_num = scaled_length * scale

    col_index = replace_index + torch.arange(replace_num)
    row_index = torch.arange(batch_size)[:, None]

    scaled_index = replace_index + torch.arange(scaled_length)
    choose_value = time_series[row_index, scaled_index, :]
    choose_value = torch.repeat_interleave(choose_value, scale, dim=1)
    time_series[row_index, col_index, :] = choose_value
    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1

    return time_series, label


def jitering_distort(time_series: torch.Tensor, subsequence_length=None, inject_rate=None):
    """
    @param time_series: (batch_size, length, 1)
    @param rate: float, the rate of distortion
    @return:

    Parameters
    ----------
    subsequence_length
    """
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    if subsequence_length is None:
        if inject_rate is None:
            rate = np.random.uniform(0.10, 0.15)
        else:
            rate = inject_rate
        replace_length = length
    else:
        if inject_rate is None:
            rate = np.random.uniform(0.1, 1)
        else:
            rate = inject_rate
        replace_length = subsequence_length

    replace_num = int(replace_length * rate)

    if replace_num == 0:
        replace_num += 1

    if replace_num >= length:
        replace_num = length - 2

    replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
    # select values to be replaced
    col_index = replace_index + torch.arange(replace_num)
    row_index = torch.arange(batch_size)[:, None]

    min_value = torch.min(time_series, dim=1, keepdim=True)[0]
    max_value = torch.max(time_series, dim=1, keepdim=True)[0]
    noise_amplitude = (max_value - min_value) * 0.05
    noise_value = torch.randn((len(row_index), replace_num, 1), device=time_series.device) * noise_amplitude
    time_series[row_index, col_index, :] = time_series[row_index, col_index, :] + noise_value

    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1

    return time_series, label


def sudden_add_drop(time_series: torch.Tensor, subsequence_length=None):
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    if subsequence_length is None:
        rate = np.random.uniform(0.10, 0.15)
        replace_length = length
    else:
        rate = np.random.uniform(0.1, 1)
        replace_length = subsequence_length

    replace_num = int(replace_length * rate)

    if replace_num == 0:
        replace_num += 1

    replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
    # select values to be replaced
    col_index = replace_index + torch.arange(replace_num)
    row_index = torch.arange(batch_size)[:, None]

    min_value = torch.min(time_series, dim=1, keepdim=True)[0]
    max_value = torch.max(time_series, dim=1, keepdim=True)[0]
    noise_amplitude = (max_value - min_value) * (0.1 + 0.5 * torch.rand(1, device=time_series.device))
    noise_value = torch.randn((len(row_index), 1, 1), device=time_series.device)
    noise_value = noise_value.repeat(1, replace_num, 1) * noise_amplitude
    time_series[row_index, col_index, :] = time_series[row_index, col_index, :] + noise_value

    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1

    return time_series, label


def frequency_distort(time_series: torch.Tensor):
    pass


def point_distort(time_series: torch.Tensor, subsequence_length=None):
    """
    Parameters
    ----------
    time_series: (batch_size, length, 1)

    Returns
    -------
    """
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    replace_num = np.random.randint(1, 10)
    replace_index = torch.randint(0, length - 2, (batch_size, replace_num))
    # select values to be replaced
    col_index = replace_index + torch.arange(1)
    row_index = torch.arange(batch_size)[:, None]

    min_value = torch.min(time_series, dim=1, keepdim=True)[0]
    max_value = torch.max(time_series, dim=1, keepdim=True)[0]
    noise_amplitude = (max_value - min_value) * 0.5
    noise_value = torch.randn((len(row_index), replace_num, 1), device=time_series.device) * noise_amplitude
    time_series[row_index, col_index, :] = time_series[row_index, col_index, :] + noise_value

    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1

    return time_series, label


def original(time_series: torch.Tensor, subsequence_length=None, inject_rate=None):
    """

    Parameters
    ----------
    time_series

    Returns
    -------

    """
    batch_size, length, _ = time_series.shape
    label = torch.zeros((batch_size, length))
    return time_series, label


def soft_raplacement(time_series: torch.Tensor, subsequence_length=None):
    """
    :param time_series:
    :param subsequence_length:
    :return:
    """
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    if subsequence_length is None:
        rate = np.random.uniform(0.1, 1)
        replace_length = length
    else:
        rate = np.random.uniform(0.1, 1)
        replace_length = subsequence_length

    replace_num = int(replace_length * rate)

    if replace_num == 0:
        replace_num += 1

    replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
    random_index = torch.randint(0, length - replace_num, (batch_size, 1))

    col_index = replace_index + torch.arange(replace_num)
    row_index = torch.arange(batch_size)[:, None]

    choose_value = time_series[row_index, col_index, :]
    random_value = time_series[row_index, random_index, :]

    weight = torch.rand((batch_size, 1, 1), device=time_series.device)
    replace_value = weight * choose_value + (1 - weight) * random_value

    time_series[row_index, col_index, :] = replace_value
    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1

    return time_series, label


def mirror_flip(time_series: torch.Tensor, subsequence_length=None, inject_rate=None):
    time_series = copy.deepcopy(time_series)
    batch_size, length, _ = time_series.shape
    if subsequence_length is None:
        if inject_rate is None:
            rate = np.random.uniform(0.10, 0.15)
        else:
            rate = inject_rate
        replace_length = length
    else:
        if inject_rate is None:
            rate = np.random.uniform(0.1, 1)
        else:
            rate = inject_rate
        replace_length = subsequence_length

    replace_num = int(replace_length * rate)

    if replace_num == 0:
        replace_num += 1

    if replace_num >= length:
        replace_num = length - 2

    replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
    random_index = torch.randint(0, length - replace_num, (batch_size, 1))

    col_index = replace_index + torch.arange(replace_num)
    row_index = torch.arange(batch_size)[:, None]

    # batch_size, replace_num, 1
    choose_value = time_series[row_index, col_index, :]
    choose_value = torch.flip(choose_value, [1])

    time_series[row_index, col_index, :] = choose_value
    label = torch.zeros((batch_size, length))
    label[row_index, col_index] = 1

    return time_series, label


# def warping_distort(time_series: torch.Tensor, subsequence_length = None):
#     """
#     Parameters
#     ----------
#     time_series: (batch_size, length, 1)
#     subsequence_length
#
#     Returns
#     -------
#     """
#     time_series = copy.deepcopy(time_series)
#     batch_size, length, _ = time_series.shape
#     if subsequence_length is None:
#         rate = np.random.uniform(0.1, 1)
#         replace_length = length
#     else:
#         rate = np.random.uniform(0.1, 1)
#         replace_length = subsequence_length
#
#     replace_num = int(replace_length * rate)
#     if replace_num == 0:
#         replace_num += 1
#
#     replace_index = torch.randint(0, length - replace_num, (batch_size, 1))
#     # select values to be replaced
#     col_index = replace_index + torch.arange(replace_num)
#     row_index = torch.arange(batch_size)[:, None]
#
#     slices = time_series[row_index, col_index, :]
#     fft_values = torch.fft.fft(slices, dim=1)
#     psd_values = torch.abs(fft_values) ** 2
#     peak_indices = torch.argsort(psd_values, dim=1)[:, -30: ]
#
#     frequencies = torch.fft.fftfreq(length, d=1)
#     frequencies = frequencies.unsqueeze(dim=0).unsqueeze(-1).repeat(batch_size, 1, 1)
#
#     frequencies = frequencies[peak_indices]
#     frequencies = torch.unique(frequencies[frequencies>0], dim=1)
#     frequencies = torch.sort(frequencies, dim=1)
#
#     return time_series, label


if __name__ == '__main__':
    x = torch.linspace(0, 2 * np.pi, 2048)
    # y_sin = torch.sin(x)
    y_cos = torch.cos(x)
    y = torch.stack([x, y_cos], dim=0)
    y = y.unsqueeze(2)
    print(y.shape)
    y_distorted, label = mirror_flip(time_series=y, subsequence_length=256)

    y = y.detach().cpu().numpy()
    y_distorted = y_distorted.detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(x, y[0], label='original')
    plt.plot(x, y_distorted[0], label='distorted')
    plt.legend()
    # plt.plot(x, y_distorted[1], label)
    # plt.plot(x, y_distorted[0], label='distorted')
    plt.show()
