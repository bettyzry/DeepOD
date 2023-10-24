import random

import gym
import numpy as np

from gym import spaces
from sample_selection.ssutil import percentile
from deepod.utils.utility import get_sub_seqs_label
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader


class ADEnv(gym.Env):
    """
    Customized environment for anomaly detection
    """

    def __init__(self, dataset: np.ndarray, clf, device='cpu', y=None, seq_len=30, stride=30,
                 num_sample=1000):
        """
        Initialize anomaly environment for DPLAN algorithm.
        :param num_sample: Number of sampling for the generator g_u
        """
        super().__init__()
        self.reward_dis = None
        self.e = 0.5
        self.device = device
        self.seq_len = seq_len
        self.stride = stride

        # Dataset infos
        self.n_samples, self.n_feature = dataset.shape
        self.x = dataset  # 原始数据
        self.train_start = np.arange(0, self.n_samples - seq_len + 1, stride)  # 训练集序列的索引标签（初始化为无重复的序列开头）
        self.train_seqs = np.array([self.x[i:i + self.seq_len] for i in self.train_start])  # 当前的训练集序列（初始化为无重复的序列）

        self.y = y
        self.train_label = get_sub_seqs_label(y, seq_len=self.seq_len, stride=stride) if y is not None else None

        # hyper parameter
        # 贪婪算法选择行动时，为了提高效率进行了采样。如果训练集少于numsample，则使用全集
        self.num_sample = min(num_sample, len(self.train_start))

        # state space:
        self.state_space = spaces.Discrete(self.n_samples)  # 状态空间（全部的训练集)

        # action space: # 0扩展，1保持，2删除
        self.action_space = spaces.Discrete(3)  # 0扩展，1保持，2删除.0扩展1删除

        # initial state
        self.state_a = None  # state in all data 当前状态在全部数据中的索引值
        self.state_t = None  # state in training data当前状态在训练集里的索引值
        self.DQN = None
        self.loss = []

        self.clf = clf
        self.init_clf()

    def init_clf(self):
        self.clf.trainsets['seqstarts0'] = self.train_start
        self.clf.n_samples, self.clf.n_features = self.train_seqs.shape[0], self.train_seqs.shape[2]
        if self.y is not None:
            self.clf.trainsets['yseq0'] = self.train_label

    def from_sa2st(self, sa):
        st = np.where(self.train_start == sa)[0][0]
        return st

    def from_st2sa(self, st):
        sa = self.train_start[st]
        return sa

    def generater(self, action, s_a, *args, **kwargs):
        if action == 0:    # expand
            next_sa = []
            if s_a - self.clf.split[0] >= 0:
                next_sa.append(s_a - self.clf.split[0])
            if s_a + self.clf.split[1] < len(self.x) - self.seq_len + 1:
                next_sa.append(s_a + self.clf.split[1])
            if s_a - self.clf.split[1] > 0:
                next_sa.append(s_a - self.clf.split[1])
            if s_a + self.clf.split[0] <= len(self.x) - self.seq_len + 1:
                next_sa.append(s_a + self.clf.split[0])
            index = np.random.choice(next_sa)
            # index = next_sa
        elif action == 1:   # save
            index = np.random.choice(self.train_start)
        else:       # delete
            index = np.random.choice(self.train_start)
        return index

    def generater_r(self, *args, **kwargs):  # 删除
        # sampling function for D_a
        index = np.random.choice(self.train_start)
        return index

    def generate_u(self, action, s_a, s_t):
        # 对状态s_t要执行action操作
        # sampling function for D_u
        # 在所有的训练集中随机采num_S个，S为采样数据in all data的索引号               # 为了效率
        S = np.random.choice(range(len(self.train_seqs)), self.num_sample)
        # calculate distance in the space of last hidden layer of DQN
        # all_x = self.train_seqs[S].append(self.x[s_a: s_a + self.seq_len])  # 提取全部采样点+当前位置，对应的数据的值
        all_x = np.concatenate((self.train_seqs[S], [self.x[s_a: s_a + self.seq_len]]), axis=0)

        all_dqn_s = self.DQN.get_latent(all_x)  # 提取数据的表征
        all_dqn_s = all_dqn_s.cpu().detach().numpy()
        dqn_s = all_dqn_s[:-1]
        dqn_st = all_dqn_s[-1]

        dist = np.linalg.norm(dqn_s - dqn_st, axis=1)  # 采样数据点与当前状态st的距离
        dist = np.average(dist, axis=1)

        # 0扩展，1保持，2删除
        if action == 0:  # 扩展该数据
            loc = np.argmin(dist)  # 找最像的
        elif action == 1:
            loc = random.randint(0, self.num_sample-1)
        else:  # action == 2 # 删除该数据
            loc = np.argmax(dist)  # 找最不像的
        state_t = S[loc]
        state_a = self.from_st2sa(state_t)
        return state_a       # 返回state_a

    # def reward_h(self, state_t, epoch, ii):  # 计算reward
    #     batch_size = 8
    #     # 根据关键参数，确定收益
    #     # x = self.train_seqs[state_t:state_t+self.clf.batch_size, :]  # 提取一个batch的数据
    #     state_a = self.from_st2sa(state_t)
    #     if state_a < int(batch_size/2):
    #         start = 0
    #         end = batch_size
    #     elif state_a > self.n_samples-int(batch_size/2):
    #         start = self.n_samples-batch_size-1
    #         end = self.n_samples
    #     else:
    #         start = state_a-int(batch_size/2)
    #         end = state_a+int(batch_size/2)
    #
    #     x = np.array([self.x[i: i + self.seq_len] for i in np.arange(start, end, 1)])
    #     loader = DataLoader(x, batch_size=batch_size, shuffle=True, drop_last=False)
    #     self.clf.net.eval()
    #     for ii, batch_x in enumerate(loader):
    #         metric = self.clf.get_importance_ICLR21(batch_x, epoch, ii)  # 直接计算all_v
    #
    #     # x = self.train_seqs[state_t, :]  # 提取一个batch的数据
    #     # batch_x = torch.tensor(x, dtype=torch.float32, device='gpu')
    #     # metric = self.clf.get_importance_ICLR21(batch_x, epoch, ii)  # 直接计算all_v
    #     self.clf.net.train()
    #     metric_torch = torch.tensor(metric, dtype=torch.float32, device='cpu')
    #     self.clf.iforest.fit(metric_torch)
    #     dis = self.clf.iforest.decision_function(metric_torch)
    #     dis = np.average(dis)
    #     return dis

        # threshold2 = percentile(self.loss, 0.2)
        # threshold8 = percentile(self.loss, 0.8)
        # if action == 0:     # 删除该点
        #     if self.loss[state_t] >= threshold8:
        #         return 1
        #     elif self.loss[state_t] <= threshold2:
        #         return -1
        #     else:
        #         return 0
        # else:
        #     if self.loss[state_t] >= threshold8:
        #         return -1
        #     elif self.loss[state_t] <= threshold2:
        #         return 1
        #     else:
        #         return 0

    def step(self, action, state_a, state_t):
        # 执行这个行动，并输出
        # store former state
        # state_a = self.state_a
        # state_t = self.state_t
        # choose generator

        state_a1 = self.generater(action, state_a)
        # g = np.random.choice([self.generater_r, self.generate_u], p=[0.5, 0.5])
        # state_a1 = g(action, state_a, state_t)  # 找到下一个要探索的点

        if state_a not in self.train_start:
            x = self.x[state_a: state_a+self.seq_len]
            x = torch.tensor(x)
            reward = self.clf.get_importance_ICLR21(x)
        else:
            # calculate the reward
            # reward = self.reward_h(state_t, -1, 0)    # 当前的行为能获得多大的收益
            reward = self.reward_dis[state_t]           # 当前的行为能获得多大的收益

        # self.state_a = state_a1
        # self.state_t = self.from_sa2st(state_a1)

        return state_a1, reward

    def reset_state(self):
        # reset the status of environment
        # the first observation is uniformly sampled from the D_u
        self.state_a = np.random.choice(self.train_start)
        self.state_t = self.from_sa2st(self.state_a)

        return self.state_a, self.state_t
