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
    datasize = int(len(train_data)/seq_len)
    onum = int(datasize*rate)
    ostarts = random.sample(range(0, datasize-1), onum)
    train_labels = np.zeros(len(train_data))
    for ostart in ostarts:
        index = random.randint(0, okinds-1)
        train_data[ostart*seq_len: (ostart+1)*seq_len] += test_seq[oseqs[index]]
        train_labels[ostart*seq_len: (ostart+1)*seq_len] = 1
    return train_data, train_labels


def insert_pollution_seq(test_data, labels, rate, seq_len):
    ori_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    oriy_seq = get_sub_seqs_label(labels, seq_len=seq_len, stride=1)

    split = 0.6
    train_num = int(len(ori_seq)*split)
    train_seq = ori_seq[:train_num]

    oseqs = np.where(oriy_seq == 1)[0]
    okinds = len(oseqs)

    ii = 0
    train_seq_o = []
    train_labels = []
    train_num_o = 3*train_num
    while len(train_seq_o) <= train_num_o:
        l = random.random()
        if l <= rate:
            oindex = random.randint(0, okinds-1)
            train_seq_o.append(ori_seq[oseqs[oindex]])
            train_labels.append(1)
        else:
            train_seq_o.append(train_seq[ii])
            train_labels.append(0)
            ii += 1
            if ii == len(train_seq):
                ii = 0
    train_seq_o = np.array(train_seq_o)
    train_labels = np.array(train_labels)

    test_data = test_data[train_num:]
    labels = labels[train_num:]
    return train_seq_o, train_labels, test_data, labels
