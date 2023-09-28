import numpy as np
from sklearn import metrics
import random


def get_sub_seqs(x_arr, seq_len=100, stride=1):
    """

    Parameters
    ----------
    x_arr: np.array, required
        input original data with shape [time_length, channels]

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    x_seqs: np.array
        Split sub-sequences of input time-series data
    """

    seq_starts = np.arange(0, x_arr.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])

    return x_seqs


def get_sub_seqs_label(y, seq_len=100, stride=1):
    """

    Parameters
    ----------
    y: np.array, required
        data labels

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    y_seqs: np.array
        Split label of each sequence
    """

    seq_starts = np.arange(0, y.shape[0] - seq_len + 1, stride)
    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    y = np.sum(ys, axis=1) / seq_len

    y_binary = np.zeros_like(y)
    y_binary[np.where(y!=0)[0]] = 1
    return y_binary


def get_sub_seqs_label2(y, seq_starts, seq_len):
    """

    Parameters
    ----------
    y: np.array, required
        data labels

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    y_seqs: np.array
        Split label of each sequence
    """

    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    y = np.sum(ys, axis=1) / seq_len

    y_binary = np.zeros_like(y)
    y_binary[np.where(y!=0)[0]] = 1
    return y_binary


def insert_pollution(train_data, test_data, labels, rate, seq_len):
    test_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    y_seqs = get_sub_seqs_label(labels, seq_len=seq_len, stride=1)
    oseqs = np.where(y_seqs == 1)[0]
    okinds = len(oseqs)
    datasize = len(train_data)
    onum = int(datasize*rate/seq_len)
    ostarts = [random.randint(0, datasize-seq_len-1) for i in range(onum)]
    train_labels = np.zeros(datasize)
    for ostart in ostarts:
        index = random.randint(0, okinds-1)
        train_data[ostart: ostart+seq_len] += test_seq[index]
        train_labels[ostart: ostart+seq_len] = 1
    return train_data, train_labels
