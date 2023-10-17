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


def insert_outlier(train, num, okind, test_label=None):
    train = pd.DataFrame(train)
    N = len(train)
    Columns = len(train.columns)
    actions = {okind: []}

    sum_num = 20
    sep = int(N / sum_num)
    factor = 4

    if okind == 'extreme':
        timestamp = 1
    else:
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

    timestamps = [(l, l + timestamp) for l in realloc]
    for n in range(Columns):
        actions[okind].append({'n': n, 'timestamps': timestamps, 'factor': factor})
    train = add_outliers(train, actions)
    labels = np.zeros(N)
    for l in realloc:
        labels[l:l + timestamp] = 1
    return train, labels


def main():
    df = pd.read_csv('/home/xuhz/dataset/5-TSdata/_processed_data/SMD/machine-1-3/machine-1-3_train.csv')[['A0', 'A1']]
    label = pd.read_csv('/home/xuhz/dataset/5-TSdata/_processed_data/SMD/machine-1-3/machine-1-3_test.csv')
    label = label[['label']].values
    columns = df.columns
    # for col in columns:
    #     plt.plot(df[col], label=col)
    # plt.legend()
    # plt.show()

    df, labels = insert_outlier(df, 20, 'variance')

    for col in columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
    # df = {'extreme': [{'n': 0, 'timestamps': [(122, 10000)], 'factor': 8}],
    #       'shift': [{'n': 0, 'timestamps': [(1000, 2000), (3000, 4000)], 'factor': -8}],
    #       'trend': [{'n': 0, 'timestamps': [(7000, 8000)], 'factor': 0.005}],      # 1000*0.005
    #       'variance': [{'n': 0, 'timestamps': [(13000, 14000)], 'factor': 10}]})
