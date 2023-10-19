# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
#
#
# # conv1 = nn.Conv1d(in_channels=19, out_channels=10, kernel_size=5, stride=1, padding=0)
# # input = torch.randn(1, 30, 19)  # [batch_size, max_len, embedding_dim]
# # input = input.permute(0, 2, 1)    # 交换维度：[batch_size, embedding_dim, max_len]
# # out = conv1(input)                # [batch_size, out_channels, n+2p-f/s+1]
# # print(out.shape)      			  # torch.Size([32, 100, 33])
#
# # conv2 = nn.Conv2d(in_channels=256, out_channels=100, kernel_size=3, stride=1, padding=0)
# # input = torch.randn(32, 35, 256)  # [batch_size, max_len, embedding_dim]
# # input = input.permute(0, 2, 1)    # 交换维度：[batch_size, embedding_dim, max_len]
# # out = conv2(input)                # [batch_size, out_channels, n+2p-f/s+1]
# # print(out.shape)      			  # torch.Size([32, 100, 33])
#
# x = torch.randn(64, 3, 9)  # [batch_size, max_len, embedding_dim]
# y = x.view(-1, 3*9)
# print(x.cpu().detach().numpy()[0])
# print(y.cpu().detach().numpy()[0])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/xuhz/dataset/5-TSdata/_processed_data/ASD/omi-5/omi-5_test.csv'
df = pd.read_csv(path)
label = df['label'].values[700: 850]
feature = df['A6'].values[700: 850]
plt.plot(feature)

outlier = np.where(label == 1)[0]

size = 30
ax = plt.gca()
for i in range(outlier[0], outlier[0]+5):
    ax.add_patch(plt.Rectangle((i, 0), 30, 1, color="r", fill=False, linewidth=1))

plt.axvspan(outlier[0], outlier[-1], ymin=0, ymax=1, color='r', alpha=0.2)


plt.show()