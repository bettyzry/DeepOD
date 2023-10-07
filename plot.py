import getpass
import random

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from scipy.spatial.distance import pdist
from deepod.metrics import ts_metrics, point_adjustment, point_adjustment_min
import seaborn as sns
from deepod.utils.utility import get_sub_seqs_label
from testbed.utils import import_ts_data_unsupervised
dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'


def zscore(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))  # 最值归一化


def plot_loss_distribution():
    fillname = 'ASD'     # 'MSL_combined'
    step = 100
    data_root = '/home/xuhz/zry/DeepOD-new/@trainsets/TcnED./%s_combined_norm0.80.csv' % fillname
    df = pd.read_csv(data_root)
    step = str(step)

    adjloss = point_adjustment(df['yseq0'].values, df['loss'+step].values)
    df['adjloss'] = adjloss


    true = df[df.yseq0 == 0]
    false = df[df.yseq0 == 1]

    # 绘制多个变量的密度分布图
    sns.kdeplot(true['loss'+step], shade=True, color="r", label='Clean')
    sns.kdeplot(false['loss'+step], shade=True, color="b", label='Polluted')
    plt.legend()
    plt.show()

    sns.kdeplot(true['adjloss'], shade=True, color="r", label='Clean')
    sns.kdeplot(false['adjloss'], shade=True, color="b", label='Polluted')
    plt.legend()
    plt.show()

    # sns.kdeplot(true['random_loss'], shade=True, color="r")
    # sns.kdeplot(false['random_loss'], shade=True, color="b")
    # plt.show()


def plot_dis_distribution():
    fillname = 'DASADS'     # 'MSL_combined'
    step = 0
    data_root = '/home/xuhz/zry/DeepOD-new/@trainsets/TcnED./%s_combined_myfunc-addo0.10.csv' % fillname

    df = pd.read_csv(data_root)
    adjdis = point_adjustment(df['yseq'+str(step)].values, df['dis'+str(step+1)].values)
    df['adjdis'] = adjdis
    adjval = point_adjustment_min(df['yseq'+str(step)].values, df['value'+str(step)].values)
    df['adjval'] = adjval
    adjnum = point_adjustment_min(df['yseq'+str(step)].values, df['num'+str(step)].values)
    df['adjnum'] = adjnum

    true = df[df.yseq0 == 0]
    false = df[df.yseq0 == 1]

    # 绘制多个变量的密度分布图
    # sns.kdeplot(true['dis'+str(step+1)], shade=True, color="r", label='Clean')
    # sns.kdeplot(false['dis'+str(step+1)], shade=True, color="b", label='Polluted')
    # plt.legend()
    # plt.show()

    eval_metrics = ts_metrics(df['yseq'+str(step)].values, df['dis'+str(step+1)].values)
    txt = ', '.join(['%.4f' % a for a in eval_metrics])
    print(txt)

    # sns.kdeplot(true['adjdis'], shade=True, color="r", label='Clean')
    # sns.kdeplot(false['adjdis'], shade=True, color="b", label='Polluted')
    # plt.legend()
    # plt.show()
    #
    # sns.kdeplot(true['value'+str(step)], shade=True, color="r", label='Clean')
    # sns.kdeplot(false['value'+str(step)], shade=True, color="b", label='Polluted')
    # plt.legend()
    # plt.show()
    #
    # sns.kdeplot(true['adjval'], shade=True, color="r", label='Clean')
    # sns.kdeplot(false['adjval'], shade=True, color="b", label='Polluted')
    # plt.legend()
    # plt.show()
    #
    # sns.kdeplot(true['num'+str(step)], shade=True, color="r", label='Clean')
    # sns.kdeplot(false['num'+str(step)], shade=True, color="b", label='Polluted')
    # plt.legend()
    # plt.show()
    #
    # sns.kdeplot(true['adjnum'], shade=True, color="r", label='Clean')
    # sns.kdeplot(false['adjnum'], shade=True, color="b", label='Polluted')
    # plt.legend()
    # plt.show()


def plot_dloss_distribution():
    fillname = 'ASD'     # 'MSL_combined'
    step = 0
    data_root = '/home/xuhz/zry/DeepOD-new/@losses/TimesNet./%s_combined_norm0-T.csv' % fillname

    step = str(step)
    data_pkg = import_ts_data_unsupervised(dataset_root,
                                           fillname, entities='FULL',
                                           combine=1)
    train_lst, test_lst, label_lst, name_lst = data_pkg
    test_data = test_lst[0]
    labels = label_lst[0]

    seq_len = 30
    stride = 1
    seq_starts = np.arange(0, test_data.shape[0] - seq_len + 1, stride)
    ori_seq = np.array([test_data[i:i + seq_len] for i in seq_starts])
    split = 0.6
    train_num = int(len(ori_seq)*split)
    test_data = test_data[train_num:]
    y = labels[train_num:]

    y_seqs = get_sub_seqs_label(y, seq_len=30, stride=1)
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
    sns.kdeplot(true['loss'], shade=True, color="r", label='Clean')
    sns.kdeplot(false['loss'], shade=True, color="b", label='Polluted')
    plt.legend()
    plt.show()

    sns.kdeplot(true['adjloss'], shade=True, color="r", label='Clean')
    sns.kdeplot(false['adjloss'], shade=True, color="b", label='Polluted')
    plt.legend()
    plt.show()

def plot1(data_root, fillname, step):
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


def plotT(data_root, step):
    df = pd.read_csv(data_root)
    label = 'yseq'+str(step)
    loss = 'dis'+str(step+1)
    new_df = pd.DataFrame()

    new_df['label'] = df[label].values
    new_df['loss'] = df[loss].values
    adjloss = point_adjustment(new_df['label'].values, new_df['loss'].values)
    new_df['adjloss'] = adjloss

    true = new_df[new_df.label == 0]
    false = new_df[new_df.label == 1]

    # 绘制多个变量的密度分布图
    sns.kdeplot(true['loss'], shade=True, color="r")
    sns.kdeplot(false['loss'], shade=True, color="b")
    plt.show()

    sns.kdeplot(true['adjloss'], shade=True, color="r")
    sns.kdeplot(false['adjloss'], shade=True, color="b")
    plt.show()


def pollution_rate():
    func = 'adjauc'
    # rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    runs = 0
    for ii, rate in enumerate(rates):
        data_root = '/home/xuhz/zry/DeepOD-new/@records/TranAD.EP.norm.%s.%s.csv' % (str(rate), str(runs))
        df = pd.read_csv(data_root)
        y = df[func].values
        plt.plot(y, label=str(rate))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(func)
    plt.title('TranAD.EP')
    plt.show()


if __name__ == '__main__':
    # plot_loss_distribution()
    # plot_dis_distribution()
    pollution_rate()
    # data_root = '/home/xuhz/zry/DeepOD-new/@trainsets/TranAD./%s_combined_myfunc0.csv' % fillname
    # plotT(data_root, step)
