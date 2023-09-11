import gym
import numpy as np

from gym import spaces
from ssutil import percentile


class ADEnv(gym.Env):
    """
    Customized environment for anomaly detection
    """

    def __init__(self, dataset: np.ndarray, seq_len=30, stride=1,
                 num_sample=1000):
        """
        Initialize anomaly environment for DPLAN algorithm.
        :param num_sample: Number of sampling for the generator g_u
        """
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride

        # Dataset infos
        self.n_samples, self.n_feature = dataset.shape
        self.x = dataset                                                                # 原始数据
        self.x_seq_index = np.arange(0, self.n_samples - seq_len + 1, stride)           # 训练集序列的索引标签（初始化为无重复的序列开头）
        self.x_seqs = np.array([self.x[i:i + self.seq_len] for i in self.x_seq_index])       # 当前的训练集序列（初始化为无重复的序列）

        # hyper parameter
        # 贪婪算法选择行动时，为了提高效率进行了采样。如果训练集少于numsample，则使用全集
        self.num_sample = min(num_sample, len(self.x_seq_index))

        # state space:
        self.state_space = spaces.Discrete(self.n_samples)                              # 状态空间（全部的训练集)

        # action space: 0 or 1
        self.action_space = spaces.Discrete(2)                                          # 0删除，1扩展

        # initial state
        self.counts = None
        self.state_a = None              # state in all data 当前状态在全部数据中的索引值
        self.state_t = None              # state in training data当前状态在训练集里的索引值
        self.DQN = None
        self.loss = []

    def from_sa2st(self, sa):
        st = np.where(self.x_seq_index == sa)[0][0]
        return st

    def from_st2sa(self, st):
        sa = self.x_seq_index[st]
        return sa

    def generater_r(self, *args, **kwargs):  # 从当前训练集中随机选择一个序列
        # sampling function for D_a
        index = np.random.choice(self.x_seq_index)
        return index

    def generate_u(self, action, s_a, s_t):
        # 对状态s_t要执行action操作
        # sampling function for D_u
        # 在所有的训练集中随机采num_S个，S为采样数据in all data的索引号               # 为了效率
        S = np.random.choice(self.x_seq_index, self.num_sample)
        # calculate distance in the space of last hidden layer of DQN
        all_x = self.x_seqs[S].append(self.x[s_t: s_t+self.seq_len])         # 提取全部采样点+当前位置，对应的数据的值

        all_dqn_s = self.DQN.get_latent(all_x)  # 提取数据的表征
        all_dqn_s = all_dqn_s.cpu().detach().numpy()
        dqn_s = all_dqn_s[:-1]
        dqn_st = all_dqn_s[-1]

        dist = np.linalg.norm(dqn_s - dqn_st, axis=1)  # 采样数据点与当前状态st的距离

        if action == 1:                                                         # 扩展该数据
            loc = np.argmin(dist)                                               # 找最像的
        else:                                                                   # 删除该数据
            loc = np.argmax(dist)                                               # 找最不像的
        return S[loc]

    def reward_h(self, action, state_t):                                        # 待修改
        # 根据梯度密度、关键参数，确定收益，被扩展的数据的loss位置、分布
        threshold2 = percentile(self.loss, 0.2)
        threshold8 = percentile(self.loss, 0.8)
        if action == 0:     # 删除该点
            if self.loss[state_t] >= threshold8:
                return 1
            elif self.loss[state_t] <= threshold2:
                return -1
            else:
                return 0
        else:
            if self.loss[state_t] >= threshold8:
                return -1
            elif self.loss[state_t] <= threshold2:
                return 1
            else:
                return 0

    def step(self, action):
        self.state_a = int(self.state_a)
        self.state_t = int(self.state_t)
        # store former state
        state_a = self.state_a
        state_t = self.state_t
        # choose generator

        # 以p的概率随机在2个采样器中选择一个函数
        g = np.random.choice([self.generater_r, self.generate_u], p=[0.5, 0.5])
        state_a1 = g(action, state_a, state_t)  # 找到下一个要探索的点
        state_t1 = self.from_sa2st(state_a1)

        # change to the next state
        self.state_t = state_t1
        self.state_a = state_a1
        self.state_a = int(self.state_a)
        self.state_t = int(self.state_t)
        self.counts += 1

        # calculate the reward
        reward = self.reward_h(action, state_t)        # 当前的行为能获得多大的收益

        # done: whether terminal or not
        done = False

        # info
        info = {"State t": state_a, "Action t": action, "State t+1": state_a1}

        return self.state_a, reward, done, info

    def reset_state(self):
        # reset the status of environment
        self.counts = 0
        # the first observation is uniformly sampled from the D_u
        self.state_a = np.random.choice(self.x_seq_index)

        return self.state_a
