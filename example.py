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

np.random.seed(1337)


# STREAM_LENGTH = 200
# N = 4
# K = 2
#
# dg = MultivariateDataGenerator(STREAM_LENGTH, N, K)
# df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)
#
# for col in df.columns:
#     plt.plot(df[col], label=col)
# plt.legend()
# plt.show()
#
# df.corr()
#
# df = dg.add_outliers({'extreme': [{'n': 0, 'timestamps': [(50,), (190,)]}],
#                       'shift':   [{'n': 1, 'timestamps': [(100, 190)]}],
#                       'trend':   [{'n': 2, 'timestamps': [(20, 150)]}],
#                       'variance':[{'n': 3, 'timestamps': [(20, 80)]}]})


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


df = pd.read_csv('/home/xuhz/dataset/5-TSdata/_processed_data/SMD/machine-1-3/machine-1-3_train.csv')[['A0']]
for col in df.columns:
    plt.plot(df[col], label=col)
plt.legend()
plt.show()


df = add_outliers(df, {
                       # 'extreme': [{'n': 0, 'timestamps': [(122, 10000)], 'factor': 4}],
                       'shift': [{'n': 0, 'timestamps': [(1000, 2000), (3000, 4000)], 'factor': 4}],
                       # 'trend': [{'n': 0, 'timestamps': [(7000, 8000)], 'factor': 4}],      # 1000*0.005
                       # 'variance': [{'n': 0, 'timestamps': [(13000, 14000)], 'factor': 4}]
                       })

for col in df.columns:
    plt.plot(df[col], label=col)
plt.legend()
plt.show()
