from sklearn.ensemble import IsolationForest
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing


def DQN_iforest(x, model):
    # iforest function on the penuli-layer space of DQN

    # get the output of penulti-layer
    latent_x = model.get_latent(x)
    latent_x = latent_x.cpu().detach().numpy()
    # calculate anomaly scores in the latent space
    iforest = IsolationForest(n_estimators=100).fit(latent_x)
    scores = -iforest.decision_function(latent_x)
    # normalize the scores
    # scores = np.array([-1 * s + 0.5 for s in scores])
    # 异常分数归一化
    _range = np.max(scores) - np.min(scores)
    norm_scores = (scores - np.min(scores)) / _range
    return norm_scores


def get_total_reward(action, dis, intrinsic_rewards, state_t, d, o, write_rew=False, a=0.5):
    if torch.is_tensor(action):
        action = action.numpy()[0][0]
    os = intrinsic_rewards[state_t]
    if write_rew:
        write_reward('./results/rewards.csv', os, dis)
    # 0扩展，1保持，2删除
    if action == 0:
        reward_e = (2 * d - dis)
        reward_i = os
    elif action == 1:
        reward_e = (2 * d - dis)
        reward_i = (2 * o - os)
    else:
        reward_e = dis
        reward_i = o
    reward = a * reward_e + (1 - a) * reward_i
    # if action == 0:             # 扩展
    #     if dis <= d:
    #         if os <= o:
    #             reward = 0
    #         else:
    #             reward = 1
    #     else:
    #         if os <= o:
    #             reward = -1
    #         else:
    #             reward = -2
    # elif action == 1:
    #     if dis <= d:
    #         if os <= o:
    #             reward = 1
    #         else:
    #             reward = 0
    #     else:
    #         if os <= o:
    #             reward = -1
    #         else:
    #             reward = -2
    # else:   # action == 2
    #     if dis <= d:
    #         if os <= o:
    #             reward = -2
    #         else:
    #             reward = -1
    #     else:
    #         if os <= o:
    #             reward = 0
    #         else:
    #             reward = 1
    return reward


def plot_roc_pr(test_set, policy_net):
    test_X, test_y = test_set[:, :-1], test_set[:, -1]
    pred_y = policy_net(test_X).detach().numpy()[:, 1]
    fpr, tpr, _ = roc_curve(test_y, pred_y)
    plt.plot(fpr, tpr)
    plt.show()

    display = PrecisionRecallDisplay.from_predictions(test_y, pred_y, name="DQN")
    _ = display.ax_.set_title("2-class Precision-Recall curve")


def percentile(N, percent, key=lambda x:x):
    # 计算分位点 eg. percentile([1,2,3,4,5], 0.1) = 1.4
    N = sorted(N)
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return round(d0+d1, 5)


def test_model(test_X, test_y, policy_net):
    policy_net.eval()
    outlier_score = policy_net(test_X).detach().cpu().numpy()[:, 1]

    roc = roc_auc_score(test_y, outlier_score)
    pr = average_precision_score(test_y, outlier_score)
    policy_net.train()
    return roc, pr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_results(pr_auc_history, roc_auc_history, dataset, path):
    pr_auc_history = np.array(pr_auc_history)
    roc_auc_history = np.array(roc_auc_history)
    pr_mean = np.mean(pr_auc_history)
    auc_mean = np.mean(roc_auc_history)
    pr_std = np.std(pr_auc_history)
    auc_std = np.std(roc_auc_history)
    line = f'{dataset},{pr_mean},{pr_std},{auc_mean},{auc_std}\n'

    with open(path, 'a') as f:
        f.write(line)


def write_reward(path, r_i, r_e):
    with open(path, 'a') as f:
        f.write(f'{r_i},{r_e},')
