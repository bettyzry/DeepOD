from collections import namedtuple, deque
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os

from ssutil import DQN_iforest, get_total_reward, test_model
from ENV import ADEnv

import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN
from RelayMemory import ReplayMemory

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'state_index', 'next_state_index'))


class DQNSS():
    """
    DPLAN agent that encapsulates the training and testing of the DQN
    """

    def __init__(self, env: ADEnv, test_X, test_Y, destination_path, device='gpu', double_dqn=True,
                 n_episodes=6, steps_per_episode=2000, max_memory=100000, eps_max=1, eps_min=0.1,
                 eps_decay=10000, hidden_size=10, learning_rate=0.25e-4, momentum=0.95,
                 min_squared_gradient=0.1, warmup_steps=100, gamma=0.99, batch_size=64,
                 target_update=5000, theta_update=2000, validation_frequency=100, weight_decay=1e-3):
        """
        Initialize the DPLAN agent
        :param env: the environment
        :param validation_set: the validation set
        :param test_set: the test set
        :param destination_path: the path where to save the model
        :param device: the device to use for training
        """
        self.double_dqn = double_dqn
        self.test_X = test_X
        self.test_Y = test_Y
        self.device = device
        self.env = env

        if not os.path.exists(destination_path):
            raise ValueError('destination path does not exist')

        self.destination_path = destination_path

        # tensor rapresentation of the dataset used in the intrinsic reward
        self.x_tensor = torch.tensor(env.x, dtype=torch.float32, device=device)

        # hyperparameters setup
        self.hidden_size = hidden_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_max
        self.EPS_END = eps_min
        self.EPS_DECAY = eps_decay
        self.LR = learning_rate
        self.momentum = momentum
        self.min_squared_gradient = min_squared_gradient
        self.num_episodes = n_episodes
        self.num_warmup_steps = warmup_steps
        self.steps_per_episode = steps_per_episode
        self.max_memory_size = max_memory
        self.target_update =target_update
        self.validation_frequency = validation_frequency
        self.theta_update = theta_update
        self.weight_decay = weight_decay

        #  n actions and n observations
        self.n_actions = env.action_space.n  # 可以执行的行动的数量
        self.n_feature = env.n_feature  # 有错？？

        #  resetting the agent
        self.reset_nets()

        # resetting agent's memory
        self.reset_memory()

        # resetting counters
        self.reset_counters()

    def reset_memory(self):
        self.memory = ReplayMemory(self.max_memory_size)

    def reset_counters(self):
        # training counters and utils
        self.num_steps_done = 0
        self.episodes_total_reward = []
        self.pr_auc_history = []
        self.roc_auc_history = []
        self.best_pr = None

    def reset_nets(self):
        # net definition
        self.policy_net = DQN(self.n_feature, self.hidden_size, self.n_actions, device=self.device).to(self.device)
        # not sure if this works
        # self.policy_net._initialize_weights()
        self.target_net = DQN(self.n_feature, self.hidden_size, self.n_actions, device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())       # 加载存储好的网络
        # set target net weights to 0
        with torch.no_grad():
            for param in self.target_net.parameters():
                param.zero_()

        # setting up the environment's DQN
        self.env.DQN = self.policy_net
        # setting up the environment's intrinsic reward as function of netwo rk's theta_e (i.e. the hidden layer)
        self.intrinsic_rewards = DQN_iforest(self.x_tensor, self.policy_net)            # 计算不同x的异常分数，即他们的孤立性

        # setting the rmsprop optimizer
        self.optimizer = optim.RMSprop(                                      # 优化器
            self.policy_net.parameters(),
            lr=self.LR,
            momentum=self.momentum,
            eps=self.min_squared_gradient,
            weight_decay=self.weight_decay
        )

    def select_action(self, state, steps_done):
        """
        Select an action using the epsilon-greedy policy
        :param state: the current state
        :param steps_done: the number of steps done
        :return: the action
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """
        Optimize the model using the replay memory
        """
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))  # 对应index的数据放到了一起，action，next_state等在一起

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # 把一个batch的当前状态输入进去，得到预测的所有Q，并获得对应action的Q，即Q(st, at)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)                  # 实际的Q

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # max(1): 返回A每一行最大值组成的一维数组
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]    # 预测的Q
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def warmup_steps(self):
        """
        Implement the warmup steps to fill the replay memory using random actions
        """
        for _ in range(self.num_warmup_steps):
            state_index = self.env.reset_state()                                    # 随机在未知数据中挑选一个点
            state = torch.tensor(self.env.x[state_index, :], dtype=torch.float32, device=self.device).unsqueeze(0)
            for _ in range(self.steps_per_episode):
                action = np.random.randint(0, self.n_actions)           # 随机挑选一个行动
                next_state_index, reward, _, _ = self.env.step(action)       # 执行这个行动
                reward = get_total_reward(reward, self.intrinsic_rewards, state_index)
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(self.env.x[next_state_index, :], dtype=torch.float32,
                                           device=self.device).unsqueeze(0)
                self.memory.push(state, torch.tensor([[action]], device=self.device), next_state, reward, state_index, next_state_index)
                # 'state', 'action', 'next_state', 'reward', 'state_index', 'next_state_index'
                state = next_state
                state_index = next_state_index

    def fit(self, reset_nets=False):
        """
        Fit the model according to the dataset and hyperparameters. The best model is obtained by using
        the best auc-pr score with the validation set.
        :param reset_nets: whether to reset the networks
        """

        # reset necessary variables
        self.reset_counters()
        self.reset_memory()
        if reset_nets:
            self.reset_nets()

        # perform warmup steps
        # self.warmup_steps()

        for i_episode in range(self.num_episodes):
            # Initialize the environment and get it's state
            reward_history = []
            state_index = self.env.reset_state()  # 随机挑选一个未知点
            #  mantain both the obervation as the dataset index and value
            # state为初始点的具体数值
            state = torch.tensor(self.env.x[state_index, :], dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in range(self.steps_per_episode):
                self.num_steps_done += 1

                # select_action encapsulates the epsilon-greedy policy
                action = self.select_action(state, self.num_steps_done)

                next_state_index, reward, _, _ = self.env.step(action.item())
                # states.append((self.env.x[observation,:],action.item()))

                reward = get_total_reward(reward, self.intrinsic_rewards, state_index, write_rew=False)

                reward_history.append(reward)
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                next_state = torch.tensor(self.env.x[next_state_index, :], dtype=torch.float32,
                                           device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward, state_index, next_state_index)

                # Move to the next state
                state = next_state
                state_index = next_state_index

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # update the target network
                if self.num_steps_done % self.target_update == 0:
                    policy_net_state_dict = self.policy_net.state_dict()
                    self.target_net.load_state_dict(policy_net_state_dict)
                # validation step
                if self.num_steps_done % self.validation_frequency == 0:
                    auc, pr = test_model(self.test_X, self.test_Y,self.policy_net)
                    self.pr_auc_history.append(pr)
                    self.roc_auc_history.append(auc)
                if self.num_steps_done % self.theta_update == 0:
                    self.intrinsic_rewards = DQN_iforest(self.x_tensor, self.policy_net)

            # because the theta^e update is equal to the duration of the episode we can update the theta^e here
            self.episodes_total_reward.append(sum(reward_history))

            # print the results at the end of the episode
            avg_reward = np.mean(reward_history)
            print('Episode: {} \t Steps: {} \t Average episode Reward: {}'.format(i_episode, t + 1, avg_reward))

        print('Complete')

    def save_model(self, model_name):
        """
        Save the model
        :param model_name: name of the model
        """
        file_path = os.path.join(self.destination_path, model_name)
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, model_name):
        """
        Save the model
        :param model_name: name of the model
        """
        file_path = os.path.join(self.destination_path, model_name)
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def show_results(self):
        """
        Show the results of the training
        """

        # plot total reward, pr auc and roc auc history in subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(self.episodes_total_reward)
        axs[0].set_title('Total reward per episode')
        axs[1].plot(self.pr_auc_history)
        axs[1].set_title('PR AUC per validation step')
        axs[2].plot(self.roc_auc_history)
        axs[2].set_title('ROC AUC per validation step')
        plt.show()

    def model_performance(self):
        """
        Test the model
        :param on_test_set: whether to test on the test set or the validation set
        """
        return test_model(self.test_X, self.test_Y, self.policy_net)