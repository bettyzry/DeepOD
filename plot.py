import getpass
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def zscore(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))  # 最值归一化


def main(fillname, step):
    df = pd.read_csv(data_root, index_col=0)

    # watch = df[['0', '1', '2', '3', '4', '50', '99']]
    df = df.fillna(0)
    zdf = df.values
    for i in range(len(df)):
        loss = df.values[i]
        loss = zscore(loss)
        zdf[i] = loss
    # index = [0, 1, 2, 3, 4]
    # watch = zdf[index, 0]
    # print(watch)

    loss = zdf[step]
    ldf = pd.DataFrame(loss)
    a = ldf.quantile([0.2, 0.8])
    print(a)
    res_freq = stats.relfreq(loss, numbins=50)  # numbins 是统计一次的间隔(步长)是多大
    pdf_value = res_freq.frequency
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    plt.bar(x, pdf_value, width=res_freq.binsize)
    plt.title(fillname + '-' + str(step))
    plt.show()
    return


if __name__ == '__main__':
    fillname = 'D-14'     # 'MSL_combined'
    step = 1
    data_root = '/home/xuhz/zry/DeepOD-new/@losses/TcnED./%s_min0.csv' % fillname
    main(fillname, step)