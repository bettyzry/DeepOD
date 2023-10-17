import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agots.agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator

np.random.seed(1337)

STREAM_LENGTH = 200
N = 4
K = 2

dg = MultivariateDataGenerator(STREAM_LENGTH, N, K)
df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)

for col in df.columns:
    plt.plot(df[col], label=col)
plt.legend()
plt.show()

df.corr()

# df = dg.add_outliers({'extreme': [{'n': 0, 'timestamps': [(50,), (190,)]}],
#                       'shift':   [{'n': 1, 'timestamps': [(100, 190)]}],
#                       'trend':   [{'n': 2, 'timestamps': [(20, 150)]}],
#                       'variance':[{'n': 3, 'timestamps': [(20, 80)]}]})


df = dg.add_outliers({'extreme': [{'n': 0, 'timestamps': [(122,)]}],
                      'shift':   [{'n': 1, 'timestamps': [(10, 30)]}],
                      'trend':   [{'n': 2, 'timestamps': [(40, 70)]}],
                      'variance':[{'n': 3, 'timestamps': [(50, 100)]}]})


for col in df.columns:
    plt.plot(df[col], label=col)
plt.legend()
plt.show()
