import getpass
import random

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from scipy.spatial.distance import pdist
from deepod.metrics import ts_metrics, point_adjustment
import seaborn as sns
from deepod.utils.utility import get_sub_seqs_label
from testbed.utils import import_ts_data_unsupervised
dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'


def zscore(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))  # 最值归一化


def plot_distribution():
    data_pkg = import_ts_data_unsupervised(dataset_root,
                                           fillname, entities='FULL',
                                           combine=1)
    train_lst, test_lst, label_lst, name_lst = data_pkg
    y = label_lst[0]
    y_seqs = get_sub_seqs_label(y, seq_len=30, stride=30)
    df = pd.read_csv(data_root)
    df = df.fillna(0)
    new_df = pd.DataFrame()

    y_seqs = y_seqs[:len(df[step].values)]
    new_df['label'] = y_seqs
    new_df['loss'] = df[step].values
    adjloss = point_adjustment(y_seqs, df[step].values)
    new_df['adjloss'] = adjloss
    random_loss = [random.random() for i in range(len(df[step].values))]
    new_df['random_loss'] = random_loss

    new_df.to_csv(data_root[:-4] + 'T.csv')
    true = new_df[new_df.label == 0]
    false = new_df[new_df.label == 1]

    # 绘制多个变量的密度分布图
    sns.kdeplot(true['loss'], shade=True, color="r")
    sns.kdeplot(false['loss'], shade=True, color="b")
    plt.show()

    sns.kdeplot(true['adjloss'], shade=True, color="r")
    sns.kdeplot(false['adjloss'], shade=True, color="b")
    plt.show()

    sns.kdeplot(true['random_loss'], shade=True, color="r")
    sns.kdeplot(false['random_loss'], shade=True, color="b")
    plt.show()


def plot1(fillname, step):
    df = pd.read_csv(data_root, index_col=0)

    # watch = df[['0', '1', '2', '3', '4', '50', '99']]
    df = df.fillna(0)
    zdf = df.values
    # for i in range(len(df)):
    #     loss = df.values[i]
    #     loss = zscore(loss)
    #     zdf[i] = loss

    loss = zdf[step]
    print(loss)
    res_freq = stats.relfreq(loss, numbins=50)  # numbins 是统计一次的间隔(步长)是多大
    pdf_value = res_freq.frequency
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    plt.bar(x, pdf_value, width=res_freq.binsize)
    # plt.title(fillname + '-' + str(step))
    # plt.show()
    plt.savefig(fillname + '-' + str(step)+'.png')
    return


if __name__ == '__main__':
    fillname = 'ASD'     # 'MSL_combined'
    step = '0'
    data_root = '/home/xuhz/zry/DeepOD-new/@key_params_num/TimesNet./%s_combined_myfunc0T.csv' % fillname
    # data_root = '/home/xuhz/zry/tsad-master/&results/loss/loss-record@TcnED_100_0_0_MSLFULL_norm/@loss_results0.csv'
    # plot1(fillname, step)
    plot_distribution()

    # x = np.random.random(1000000)
    # y = np.random.random(1000000)
    # x2 = np.random.randint(2, size=1000000)
    # y2 = np.random.randint(2, size=1000000)
    #
    # t1 = time.time()
    # f1_score(x2, y2)
    # t2 = time.time()
    # pdist(np.vstack([x, y]), 'cosine')
    # t3 = time.time()
    # print(t2-t1)
    # print(t3-t2)
