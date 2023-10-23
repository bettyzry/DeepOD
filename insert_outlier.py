import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agots.agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator

from agots.agots.multivariate_generators.multivariate_extreme_outlier_generator import \
    MultivariateExtremeOutlierGenerator
from agots.agots.multivariate_generators.multivariate_shift_outlier_generator import MultivariateShiftOutlierGenerator
from agots.agots.multivariate_generators.multivariate_trend_outlier_generator import MultivariateTrendOutlierGenerator
from agots.agots.multivariate_generators.multivariate_variance_outlier_generator import \
    MultivariateVarianceOutlierGenerator


def add_outliers(data, config):
    N = len(data.columns)
    STREAM_LENGTH = len(data)
    """Adds outliers based on the given configuration to the base line

     :param config: Configuration file for the outlier addition e.g.
     {'extreme': [{'n': 0, 'timestamps': [(3,)]}],
      'shift':   [{'n': 3, 'timestamps': [(4,10)]}]}
      would add an extreme outlier to time series 0 at timestamp 3 and a base shift
      to time series 3 between timestamps 4 and 10
     :return:
     """
    OUTLIER_GENERATORS = {'extreme': MultivariateExtremeOutlierGenerator,
                          'shift': MultivariateShiftOutlierGenerator,
                          'trend': MultivariateTrendOutlierGenerator,
                          'variance': MultivariateVarianceOutlierGenerator}

    generator_keys = []

    # Validate the input
    for outlier_key, outlier_generator_config in config.items():
        assert outlier_key in OUTLIER_GENERATORS, 'outlier_key must be one of {} but was'.format(OUTLIER_GENERATORS,
                                                                                                 outlier_key)
        generator_keys.append(outlier_key)
        for outlier_timeseries_config in outlier_generator_config:
            n, timestamps = outlier_timeseries_config['n'], outlier_timeseries_config['timestamps']
            assert n in range(N), 'n must be between 0 and {} but was {}'.format(N - 1, n)
            for timestamp in list(sum(timestamps, ())):
                assert timestamp in range(
                    STREAM_LENGTH), 'timestamp must be between 0 and {} but was {}'.format(STREAM_LENGTH, timestamp)

    df = data
    if data.shape == (0, 0):
        raise Exception('You have to first compute a base line by invoking generate_baseline()')
    for generator_key in generator_keys:
        for outlier_timeseries_config in config[generator_key]:
            n, timestamps = outlier_timeseries_config['n'], outlier_timeseries_config['timestamps']
            generator_args = dict(
                [(k, v) for k, v in outlier_timeseries_config.items() if k not in ['n', 'timestamps']])
            generator = OUTLIER_GENERATORS[generator_key](timestamps=timestamps, **generator_args)
            df[df.columns[n]] += generator.add_outliers(data[data.columns[n]])

    assert not df.isnull().values.any(), 'There is at least one NaN in the generated DataFrame'
    return df


def insert_outlier(dataset, train, num, okind, test_label=None):
    # extreme,shift,trend,variance
    # OUTLIER_GENERATORS = {'extreme': MultivariateExtremeOutlierGenerator,
    #                       'shift': MultivariateShiftOutlierGenerator,
    #                       'trend': MultivariateTrendOutlierGenerator,
    #                       'variance': MultivariateVarianceOutlierGenerator}

    train = pd.DataFrame(train)
    N = len(train)
    Columns = len(train.columns)
    actions = {okind: []}

    factor = 2
    if okind == 'extreme':
        timestamp = 1
        rate = num/100      # 污染率
        outliernum = int(N*rate)
        loc = pd.read_csv('loc.csv', index_col=0)
        loc = loc[dataset].values
        realloc = loc[:outliernum]
        labels = np.zeros(N)
        timestamps = [(l, ) for l in realloc]
        for l in realloc:
            labels[l] = 1
    else:
        sum_num = 20
        sep = int(N / sum_num)
        if test_label is not None:
            splits = np.where(test_label[1:] != test_label[:-1])[0] + 1
            is_anomaly = test_label[0] == 1
            outlier_length = []
            if is_anomaly:
                splits = splits[1:]
            for ii in range(1, len(splits), 2):
                outlier_length.append(splits[ii] - splits[ii - 1])
            timestamp = int(np.average(outlier_length))
        else:
            timestamp = min(1000, int(sep / 10))
        lists = np.array([0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19])
        loc = np.array([i for i in range(1000, N - timestamp, sep)])
        realloc = [i for i in loc[lists[:num]]]
        labels = np.zeros(N)
        for l in realloc:
            labels[l:l + timestamp] = 1
        timestamps = [(l, l + timestamp) for l in realloc]
    for n in range(Columns):
        actions[okind].append({'n': n, 'timestamps': timestamps, 'factor': factor})
    train = add_outliers(train, actions)
    return train.values, labels


from testbed.utils import import_ts_data_unsupervised
import getpass


def prepare():
    dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
    datasets = ['SMD', 'MSL', 'SMAP', 'SWaT_cut', 'ASD', 'DASADS', 'PUMP', 'UCR_natural_heart_vbeat', 'UCR_natural_heart_vbeat2']
    df = {}
    for dataset in datasets:
        data_pkg = import_ts_data_unsupervised(dataset_root,
                                    dataset, entities='FULL',
                                    combine=1)
        train_lst, test_lst, label_lst, name_lst = data_pkg
        for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
            N = len(train_data)
            sum_sample = int(N * 0.2)
            l = random.sample(list(np.arange(0, N)), sum_sample)
            df[dataset] = l
    df = pd.DataFrame.from_dict(df, orient='index').transpose()
    df.to_csv('loc.csv')


def test():
    # extreme,shift,trend,variance
    dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
    datasets = ['SMAP']
    for dataset in datasets:
        data_pkg = import_ts_data_unsupervised(dataset_root,
                                    dataset, entities='FULL',
                                    combine=1)
        train_lst, test_lst, label_lst, name_lst = data_pkg
        for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
            plt.plot(train_data[:, 0])
            plt.show()
            # train_data1, train_label = insert_outlier(dataset, train_data, 5, 'variance', test_label=None)
            # plt.plot(train_data1[:, 0])
            # plt.show()
            # train_data2, train_label = insert_outlier(dataset, train_data, 10, 'variance', test_label=None)
            # plt.plot(train_data2[:, 0])
            # plt.show()
            # train_data3, train_label = insert_outlier(dataset, train_data, 15, 'variance', test_label=None)
            # plt.plot(train_data3[:, 0])
            # plt.show()
            # train_data4, train_label = insert_outlier(dataset, train_data, 20, 'variance', test_label=None)
            # plt.plot(train_data4[:, 0])
            # plt.show()


if __name__ == '__main__':
    # prepare()
    test()
    #, 'MSL', 'SMAP', 'SWaT_cut', 'ASD', 'DASADS', 'PUMP', 'UCR_natural_heart_vbeat', 'UCR_natural_heart_vbeat2'
    # df = {
    #     # 'extreme': [{'n': 0, 'timestamps': [(122, 10000)], 'factor': 4}],
    #     #   'shift': [{'n': 0, 'timestamps': [(10000, 20000), (30000, 40000)], 'factor': 4}],
    #       # 'trend': [{'n': 0, 'timestamps': [(70000, 75000)], 'factor': 4}],      # 1000*0.005
    #       'variance': [{'n': 0, 'timestamps': [(9500, 10000)], 'factor': 2}]
    # }
    # dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
    # datasets = ['ASD']
    # for dataset in datasets:
    #     data_pkg = import_ts_data_unsupervised(dataset_root,
    #                                 dataset, entities='FULL',
    #                                 combine=1)
    #     train_lst, test_lst, label_lst, name_lst = data_pkg
    #     for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
    #         train_data = add_outliers(pd.DataFrame(train_data),df)
    #         plt.plot(train_data.values[:, 0])
    #         plt.show()
