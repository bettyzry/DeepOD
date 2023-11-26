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
from sklearn.preprocessing import normalize
dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'


def zscore(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))  # 最值归一化


def plot_loss_distribution():
    fillname = 'PUMP'     # 'MSL_combined'
    step = 0
    data_root = '/home/xuhz/zry/DeepOD-new/@trainsets/TcnED./%s_norm00.csv' % fillname
    df = pd.read_csv(data_root)
    step = str(step)

    adjloss = point_adjustment(df['yseq0'].values, df['loss'+step].values)
    df['adjloss'] = adjloss


    true = df[df.yseq0 == 0]
    false = df[df.yseq0 == 1]

    # 绘制多个变量的密度分布图
    sns.kdeplot(true['loss'+step], shade=True, color="b", label='Clean')
    sns.kdeplot(false['loss'+step], shade=True, color="r", label='Polluted')
    plt.legend()
    plt.show()

    sns.kdeplot(true['adjloss'], shade=True, color="b", label='Clean')
    sns.kdeplot(false['adjloss'], shade=True, color="r", label='Polluted')
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
    fontsize = 18
    plt.figure(dpi=300, figsize=(8,6))
    # 改变文字大小参数-fontsize
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    func = 'adjf1'
    # rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    rates = [0.2, 0.4, 0.6, 0.8]
    runs = 0
    name = 'DASADS'
    for ii, rate in enumerate(rates):
        # /home/xuhz/zry/DeepOD-new/@records/TranAD.SMAP_combined.norm.0.0.csv
        # data_root = '/home/xuhz/zry/DeepOD-new/@records/TranAD.%s.norm.%s.%s.csv' % (name, str(rate), str(runs))
        data_root = '/home/xuhz/zry/DeepOD-new/plotsource/per_by_rate/TranAD.DASADS_combined.norm.%s.0.csv' % str(rate)
        df = pd.read_csv(data_root)
        y = df[func].values
        plt.plot(y, label=str(int(rate*100))+'% symnoise')
    plt.legend(fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('F1', fontsize=fontsize)
    # plt.title('')
    # plt.show()
    plt.savefig('./plotsource/f1_rate.eps', dpi=300)
    plt.savefig('./plotsource/f1_rate.png', dpi=300)


def plot_hotmap():
    # 数据-参数重要性 热图
    data_root = '/home/xuhz/zry/DeepOD-new/@g_detail/1.csv'
    df = pd.read_csv(data_root, index_col=0)
    namelist = [str(i) for i in range(1, 100)]
    df = df.rename(columns={'0': 'label'})
    true = df[df.label == 0][namelist]
    false = df[df.label == 1][namelist]
    sns.heatmap(data=true)
    plt.show()
    sns.heatmap(data=false)
    plt.show()
    return


def plot_param_distribution():
    data_root = '/home/xuhz/zry/DeepOD-new/@g_detail/TcnED-PUMA-dL-ori/2.csv'
    df = pd.read_csv(data_root, index_col=0)
    df = df.rename(columns={'0': 'label'})

    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))      # 按列归一化

    values = df.iloc[:, 1:].values

    # mean = np.mean(values, axis=0)
    # # metric = metric / mean      # 按列归一化
    # values = np.divide(values, mean, out=np.zeros_like(values, dtype=np.float64), where=mean != 0)
    values = normalize(values, axis=1, norm='l2')   # 对metric按行进行归一化

    k100 = []
    for ii, value in enumerate(values):
        index = np.argsort(value)[:100]       # 最关键的100个参数
        k100.append(index)
    newdf = pd.DataFrame(k100)
    true = newdf[df.label == 0].values
    false = newdf[df.label == 1].values

    true = np.concatenate(true)
    false = np.concatenate(false)

    plt.hist(true)
    plt.show()
    plt.hist(false)
    plt.show()


def plot_singledata_param():
    from matplotlib import pyplot
    plt.style.use('seaborn-whitegrid')
    palette = pyplot.get_cmap('Set1')

    model = 'TcnED'
    data = 'ASD'
    truelist = []
    falselist = []
    dtindex = np.array([6, 29, 74, 104, 120, 123, 134, 138, 168, 200])
    dfindex = np.array([9, 10, 14, 16, 21, 23, 27, 30, 38, 39])
    title = ''
    for i in range(1, 10):
        data_root = '/home/xuhz/zry/DeepOD-new/@g_detail/%s-%s-ICLR21-ori/%d.csv' % (model, data, i)
        df = pd.read_csv(data_root, index_col=0)
        df = df.rename(columns={'0': 'label'})
        df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))      # 按列归一化

        true = df[df.label == 0].values[dtindex]
        false = df[df.label == 1].values[dfindex]
        true = np.mean(true, axis=1)
        false = np.mean(false, axis=1)
        truelist.append(true)
        falselist.append(false)

    truelist = np.array(truelist)
    falselist = np.array(falselist)

    iters = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    color=palette(0)
    avgt=np.mean(truelist,axis=1)
    stdt=np.std(truelist,axis=1)
    r1 = list(map(lambda x: x[0]-x[1], zip(avgt, stdt))) #上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avgt, stdt))) #下方差
    plt.plot(iters, avgt, label="Clean", color=color, linewidth=3.0)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

    color=palette(1)
    avgf=np.mean(falselist,axis=1)
    stdf=np.std(falselist,axis=1)
    r1 = list(map(lambda x: x[0]-x[1], zip(avgf, stdf)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avgf, stdf)))#下方差
    plt.plot(iters, avgf, label="Polluted",color=color, linewidth=3.0)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

    plt.legend()
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Parameter Importance', fontsize=15)
    plt.title(title)
    plt.show()

    result = np.concatenate([avgt, stdt, avgf, stdf])

    result_file = '/home/xuhz/zry/DeepOD-new/ModelParam.csv'
    dfp = pd.read_csv(result_file, index_col=0)
    dfp['%s-%s' % (model, data)] = result
    dfp.to_csv(result_file, index=False)
    return


def plot_pollute():
    dataset = '/home/xuhz/zry/DeepOD-new/polluted.csv'
    df = pd.read_csv(dataset).values[:, 1:]
    ts = [0, 5, 10, 15, 20]

    plt.figure(figsize=(16, 5))

    model = ['ASD', 'MSL', 'SMAP', 'SMD', 'SWaT',
             'PUMP', 'DASADS', 'Fault', 'Gait', 'Heart Sbeat']
    func = ['Norm', 'RODA', 'ICLR21', 'Arxiv22']

    for i in range(10):
        plt.subplot(2, 5, i+1)
        index = np.array([10 * j + i for j in range(5)])
        ASD = df[index, :]
        for k in range(4):
            avg = ASD[:, k * 2]
            std = ASD[:, k * 2 + 1]
            r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  # 上方差
            r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  # 下方差
            plt.plot(ts, avg, linewidth=3.5, label=func[k])
            plt.fill_between(ts, r1, r2, alpha=0.1)
        # plt.ylim(0.6, 0.85)
        plt.title(model[i], fontsize=12)
        if i == 0 or i == 5:
            plt.ylabel('F1-Score', fontsize=12)

    plt.legend(ncol=2, loc='upper center')  # 图例的位置，bbox_to_anchor=(0.5, 0.92),

    plt.tight_layout()
    plt.show()


def test():
    import os
    dataset_root_DC = f'/home/{getpass.getuser()}/dataset/5-TSdata/_DCDetector/'
    data = 'SMAP'
    train = np.load(os.path.join(dataset_root_DC, data, data + '_train.npy'))
    label = np.load(os.path.join(dataset_root_DC, data, data + '_test_label.npy'))[:38000]
    test = np.load(os.path.join(dataset_root_DC, data, data + '_test.npy'))[:38000]
    index = 0
    plt.plot(train[:, index])
    plt.show()
    plt.plot(test[:, index])
    plt.plot(label*np.max(test[:, index]))
    plt.show()


if __name__ == '__main__':
    # test()
    # plot_loss_distribution()
    # plot_dis_distribution()
    # pollution_rate()    # 随污染率增加f1的变化
    plot_pollute()          # 不同方法在不同污染率下的性能
    # plot_hotmap()
    # plot_param_distribution()
    # plot_singledata_param()
    # data_root = '/home/xuhz/zry/DeepOD-new/@trainsets/TranAD./%s_combined_myfunc0.csv' % fillname
    # plotT(data_root, step)
