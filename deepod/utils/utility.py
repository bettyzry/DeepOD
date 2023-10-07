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
    y = np.sum(ys, axis=1)

    y_binary = np.zeros_like(y)
    y_binary[np.where(y != 1)[0]] = 2
    y_binary[np.where(y >= seq_len/3)[0]] = 1
    y_binary[np.where(y == 0)[0]] = 0
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


def split_pollution(test_data, labels):
    num = int(0.6*len(test_data))
    train_data, train_labels = test_data[:num], labels[:num]
    test_data, labels = test_data[num:], labels[num:]
    return train_data, train_labels, test_data, labels


def insert_pollution_new(test_data, labels, rate):
    # 将一个序列的outlier插入
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    splits = np.concatenate([[0], splits])
    is_anomaly = labels[0] == 1
    data_splits = [test_data[sp: splits[ii+1]] for ii, sp in enumerate(splits[:-1])]
    outliers = [data_splits[ii] for ii, sp in enumerate(data_splits) if ii % 2 != is_anomaly]

    split = 0.6
    train_num = int(len(test_data)*split)
    train_data = test_data[:train_num]
    train_l = labels[:train_num]

    ii = 0
    train_data_o = np.array([])
    train_labels = np.array([])
    train_num_o = train_num
    while len(train_data_o) <= train_num_o:
        N = len(train_data_o)
        No = sum(train_labels)
        if N == 0:
            oindex = random.randint(0, len(outliers)-1)
            train_data_o = outliers[oindex]
            train_labels = np.ones(len(outliers[oindex]))
        elif No/N <= rate:
            oindex = random.randint(0, len(outliers)-1)
            train_data_o = np.insert(train_data_o, N, outliers[oindex], axis=0)
            train_labels = np.concatenate([train_labels, np.ones(len(outliers[oindex]))])
        else:       # 插入异常,把一整个序列装进去
            while train_l[ii % len(train_l)] == 1:
                ii += 1
            train_data_o = np.insert(train_data_o, N, train_data[ii], axis=0)
            train_labels = np.concatenate([train_labels, [0]])
            ii += 1

    test_data = test_data[train_num:]
    labels = labels[train_num:]
    return train_data_o, train_labels, test_data, labels


def insert_pollution_seq(test_data, labels, rate, seq_len):
    # 插入一长序列
    ori_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    oriy_seq = get_sub_seqs_label(labels, seq_len=seq_len, stride=1)

    split = 0.6
    train_num = int(len(ori_seq)*split)
    train_seq = ori_seq[:train_num]
    train_l = oriy_seq[:train_num]

    oseqs = np.where(train_l == 1)[0]

    ii = 0
    jj = 0
    train_seq_o = []
    train_labels = []
    train_num_o = train_num
    while len(train_seq_o) <= train_num_o:
        l = random.random()
        if l <= rate:       # 插入异常
            train_seq_o.append(ori_seq[oseqs[jj % len(oseqs)]])
            train_labels.append(1)
            jj += 7
        else:               # 插入正常
            while train_l[ii % len(train_l)] > 0:
                ii += 1
            train_seq_o.append(train_seq[ii % len(train_l)])
            train_labels.append(0)
            ii += 7
    train_seq_o = np.array(train_seq_o)
    train_labels = np.array(train_labels)

    test_data = test_data[train_num:]
    labels = labels[train_num:]
    return train_seq_o, train_labels, test_data, labels