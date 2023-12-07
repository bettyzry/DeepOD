import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import getpass
import pandas as pd
import glob
import os


def example():
    # 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，
    # 所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点）
    N = 1400  # 设置1400个采样点
    x = np.linspace(0, 1, N)  # 将0到1平分成1400份

    # 设置需要采样的信号，频率分量有0，200，400和600
    y = 2 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(
        2 * np.pi * 400 * x) + 2 * np.sin(2 * np.pi * 600 * x) + 10  # 构造一个演示用的组合信号

    # plt.plot(x, y)
    # plt.title('OriWave')
    # plt.show()

    fft_y = fft(y)  # 使用快速傅里叶变换，得到的fft_y是长度为N的复数数组

    x = np.arange(N)  # 频率个数 （x的取值涉及到横轴的设置，这里暂时忽略，在第二节求频率时讲解）

    plt.plot(x, fft_y, 'black')
    plt.title('FFT', fontsize=9, color='black')
    plt.ylim(-4000, 16000)

    plt.show()


def import_ts_data_unsupervised(data_root1, data_root2, data, entities=None, combine=False):
    if False:
        pass
    else:
        if data[:3] == 'UCR':
            combine = False
        else:
            combine = combine

        if type(entities) == str:
            entities_lst = entities.split(',')
        elif type(entities) == list:
            entities_lst = entities
        else:
            raise ValueError('wrong entities')

        name_lst = []
        train_lst = []
        test_lst = []
        label_lst = []

        if len(glob.glob(os.path.join(data_root1, data) + '/*.csv')) == 0:
            machine_lst = os.listdir(data_root1 + data + '/')
            for m in sorted(machine_lst):
                if entities != 'FULL' and m not in entities_lst:
                    continue
                train_path = glob.glob(os.path.join(data_root1, data, m, '*train*.csv'))
                test_path = glob.glob(os.path.join(data_root1, data, m, '*test*.csv'))

                assert len(train_path) == 1 and len(test_path) == 1, f'{m}'
                train_path, test_path = train_path[0], test_path[0]

                train_df = pd.read_csv(train_path, sep=',', index_col=0)
                test_df = pd.read_csv(test_path, sep=',', index_col=0)
                labels = test_df['label'].values
                train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

                # normalization
                train, test = train_df.values, test_df.values
                # train, test = train_df.values, test_df.values

                train_lst.append(train)
                test_lst.append(test)
                label_lst.append(labels)
                name_lst.append(data+'-'+m)

            if combine:
                train_lst = [np.concatenate(train_lst)]
                test_lst = [np.concatenate(test_lst)]
                label_lst = [np.concatenate(label_lst)]
                name_lst = [data + '_combined']

        else:
            train_df = pd.read_csv(f'{data_root1}{data}/{data}_train.csv', sep=',', index_col=0)
            test_df = pd.read_csv(f'{data_root1}{data}/{data}_test.csv', sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)
            train, test = train_df.values, test_df.values

            train_lst.append(train)
            test_lst.append(test)
            label_lst.append(labels)
            name_lst.append(data)

        # for ii, train in enumerate(train_lst):
        #     test = test_lst[ii]
        #     train, test = data_standardize(train, test)
        #     train_lst[ii] = train
        #     test_lst[ii] = test

    return train_lst, test_lst, label_lst, name_lst


def main():
    dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
    dataset_root_DC = f'/home/{getpass.getuser()}/dataset/5-TSdata/_DCDetector/'
    data = 'MSL'
    entities = 'C-1'
    entity_combined = False
    data_pkg = import_ts_data_unsupervised(dataset_root, dataset_root_DC,
                                           data, entities=entities,
                                           combine=entity_combined)

    train_lst, test_lst, label_lst, name_lst = data_pkg
    for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):

        splits = np.where(labels[1:] != labels[:-1])[0] + 1
        splits = np.concatenate([[0], splits])
        is_anomaly = labels[0] == 1
        data_splits = [test_data[sp: splits[ii + 1]] for ii, sp in enumerate(splits[:-1])]
        outliers = [sp for ii, sp in enumerate(data_splits) if ii % 2 != is_anomaly]
        normals = [sp for ii, sp in enumerate(data_splits) if ii % 2 == is_anomaly]
        length = np.array([len(o) for o in outliers])
        fontsize = 15
        for outlier in outliers:
            if len(outlier) > 100:
                index = 0
                y = outlier
                y = y[:, index]
                x = range(len(y))
                fft_y = fft(y)  # 使用快速傅里叶变换，得到的fft_y是长度为N的复数数组
                # plt.plot(x, y)
                # plt.title(dataset_name+'ori-outlier')
                # plt.show()
                plt.figure(figsize=(8, 3))
                plt.subplot(122)
                plt.plot(x, fft_y)
                # plt.ylim(-10,10)
                plt.title('Spectrum Diagram of Abnormal Data', fontsize=fontsize)
                # plt.show()
                for normal in normals:
                    if len(normal) >= len(y):
                        n = normal[:len(y)]
                        n = n[:, index]
                        fft_n = fft(n)
                        # plt.title(dataset_name+'ori-norm')
                        # plt.plot(x, n)
                        # plt.show()
                        plt.subplot(121)
                        plt.plot(x, fft_n)
                        # plt.ylim(-10,10)
                        plt.title('Spectrum Diagram of Normal Data', fontsize=fontsize)
                        plt.savefig('./plotsource/FFT.png', dpi=300)
                        plt.show()
                        break
                break
            else:
                print("%s=%d" % (dataset_name, max(length)))


if __name__ == '__main__':
    # example()
    main()