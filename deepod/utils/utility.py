import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt


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
    y_binary[np.where(y != 0)[0]] = 1
    # y_binary[np.where(y != 0)[0]] = 1
    # y_binary[np.where(y >= seq_len/3)[0]] = 2
    # y_binary[np.where(y >= 2*seq_len/3)[0]] = 3
    # y_binary[np.where(y == seq_len)[0]] = 4
    # y_binary[np.where(y == 0)[0]] = 0
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
    # y_binary[np.where(y != 0)[0]] = 1
    # y_binary[np.where(y >= seq_len/3)[0]] = 2
    # y_binary[np.where(y >= 2*seq_len/3)[0]] = 3
    # y_binary[np.where(y == seq_len)[0]] = 4
    # y_binary[np.where(y == 0)[0]] = 0
    return y_binary


def insert_pollution(train_data, test_data, labels, rate, seq_len):
    test_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    y_seqs = get_sub_seqs_label(labels, seq_len=seq_len, stride=1)
    oseqs = np.where(y_seqs == 1)[0]
    okinds = len(oseqs)
    datasize = int(len(train_data)/seq_len)
    rate = rate/100
    onum = int(datasize*rate)
    ostarts = random.sample(range(0, datasize-1), onum)
    train_labels = np.zeros(len(train_data))
    for ostart in ostarts:
        index = random.randint(0, okinds-1)
        train_data[ostart*seq_len: (ostart+1)*seq_len] = test_seq[oseqs[index]]
        train_labels[ostart*seq_len: (ostart+1)*seq_len] = 1
    plt.plot(train_data[:, 4])
    plt.show()
    return train_data, train_labels


def split_pollution(test_data, labels):
    num = int(0.6*len(test_data))
    train_data, train_labels = test_data[:num], labels[:num]
    test_data, labels = test_data[num:], labels[num:]
    return train_data, train_labels, test_data, labels


def insert_pollution_new(train_data, test_data, labels, rate):
    # plt.plot(train_data[:, 0])
    # plt.show()
    # 将一个序列的outlier插入
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    splits = np.concatenate([[0], splits])
    is_anomaly = labels[0] == 1
    data_splits = [test_data[sp: splits[ii+1]] for ii, sp in enumerate(splits[:-1])]
    outliers = [sp for ii, sp in enumerate(data_splits) if ii % 2 != is_anomaly]

    length = [len(o) for o in outliers]
    timestamp = int(np.average(length))                # 平均异常长度
    train_labels = np.zeros(len(train_data))
    rate = rate/100
    Onum = int(len(train_data)*rate)
    N = int(Onum/timestamp)+1   # 总异常数
    sep = int(len(train_data)/N)
    # N = int(sumN/0.2*rate)                          # 本次插入的异常数目
    # loc = np.array([i for i in range(timestamp, N - timestamp, sep)])
    start = timestamp
    okinds = len(outliers)
    count = 0
    for i in range(N):
        train_data[start: len(outliers[count])+start] += outliers[count]
        train_labels[start: len(outliers[count])+start] = 1
        count += 1
        count = count % okinds
        start += sep

    # plt.plot(train_data[:, 0])
    # plt.plot(train_labels)
    # plt.show()
    #
    # plt.plot(test_data[:, 0])
    # plt.plot(labels)
    # plt.show()
    return train_data, train_labels


def insert_pollution_seq(test_data, labels, rate, seq_len):
    # 插入一长序列
    ori_seq = get_sub_seqs(test_data, seq_len=seq_len, stride=1)
    oriy_seq = get_sub_seqs_label2(labels, seq_len=seq_len, stride=1)

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