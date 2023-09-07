import gym
import time
import numpy as np

from gym import spaces


class ADEnv(gym.Env):
    """
    Customized environment for anomaly detection
    """

    def __init__(self, dataset: np.ndarray, sampling_Du=1000, prob_au=0.5, label_normal=0, label_anomaly=1,
                 name="default"):
        """
        Initialize anomaly environment for DPLAN algorithm.
        :param dataset: Input dataset in the form of 2-D array. The Last column is the label.
        :param sampling_Du: Number of sampling on D_u for the generator g_u
        :param prob_au: Probability of performing g_a.
        :param label_normal: label of normal instances
        :param label_anomaly: label of anomaly instances
        """
        super().__init__()
        self.name = name

        # hyperparameters:
        self.num_S = sampling_Du  # 在正常数据上采样的数量
        self.normal = label_normal
        self.anomaly = label_anomaly
        self.prob = prob_au

        # Dataset infos: D_a and D_u
        self.m, self.n = dataset.shape
        self.n_feature = self.n - 1  # 特征量
        self.n_samples = self.m  # 数据量
        self.x = dataset[:, :self.n_feature]  # 原始数据
        self.y = dataset[:, self.n_feature]  # 原始标签
        self.dataset = dataset
        self.index_u = np.where(self.y == self.normal)[0]  # 标签为未知的数据的索引号
        self.index_a = np.where(self.y == self.anomaly)[0]  # 标签为异常的数据的索引号
        self.index_n = np.where(self.y == 2)[0]  # 标签为正常的数据的索引号           # 有错

        # observation space:
        self.observation_space = spaces.Discrete(self.m)  # 大小为m的观测空间

        # action space: 0 or 1
        self.action_space = spaces.Discrete(2)  # 大小为2的行为空间

        # initial state
        self.counts = None
        self.state = None
        self.DQN = None

    def generater_a(self, *args, **kwargs):  # 随机选择一个异常数据
        # sampling function for D_a
        index = np.random.choice(self.index_u)

        return index

    def generater_n(self, *args, **kwargs):  # 随机选择一个正常数据
        # sampling function for D_n
        index = np.random.choice(self.index_u)

        return index

    def generate_u(self, action, s_t):
        # acton: 行动，s_t: 当前状态？
        # sampling function for D_u
        S = np.random.choice(self.index_u, self.num_S)  # 在所有的正常数据中随机采num_S个        # 处于效率考虑，在子样本上选择，而非全集
        # calculate distance in the space of last hidden layer of DQN
        all_x = self.x[np.append(S, s_t)]  # 提取全部采样点+当前位置对应的数据的值

        all_dqn_s = self.DQN.get_latent(all_x)  # 提取数据的表征
        all_dqn_s = all_dqn_s.cpu().detach().numpy()
        dqn_s = all_dqn_s[:-1]
        dqn_st = all_dqn_s[-1]

        dist = np.linalg.norm(dqn_s - dqn_st, axis=1)  # 采样数据点与当前状态st的距离

        if action == 1:  # 找距离当前状态最小的，a0将给定观测点标记为正常
            loc = np.argmin(dist)
        elif action == 0:  # 找距离当前状态最大的，a1将给定观测点标记为异常
            loc = np.argmax(dist)
        index = S[loc]  # 最终选中的点

        return index

    def reward_h(self, action, s_t):
        # Anomaly-biased External Handcrafted Reward Function h
        if (action == 1) & (s_t in self.index_a):  # 行动为将给定点标记为异常，并且该点确实在异常列表中
            return 1
        elif (action == 0) & (s_t in self.index_n):  # 行动为将给定点标记为正常，并且该点确实在正常列表中
            return 1
        elif (action == 0) & (s_t in self.index_u):  # 行动为将给定点标记为正常，但该点在未知列表中
            return 0
        elif (action == 1) & (s_t in self.index_u):  # 行动为将给定点标记为异常，但该点在未知列表中
            return -0.5
        return -1

    def step(self, action):
        self.state = int(self.state)
        # store former state
        s_t = self.state
        # choose generator

        # 以p的概率随机在三个采样器中选择一个函数
        g = np.random.choice([self.generater_a, self.generate_u, self.generater_n], p=[0.4, 0.2, 0.4])
        s_tp1 = g(action, s_t)  # 找到下一个要探索的点

        # change to the next state
        self.state = s_tp1
        self.state = int(self.state)
        self.counts += 1

        # calculate the reward
        reward = self.reward_h(action, s_t)

        # done: whether terminal or not
        done = False

        # info
        info = {"State t": s_t, "Action t": action, "State t+1": s_tp1}

        return self.state, reward, done, info

    def reset(self):
        # reset the status of environment
        self.counts = 0
        # the first observation is uniformly sampled from the D_u
        self.state = np.random.choice(self.index_u)

        return self.state
