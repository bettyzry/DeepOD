# -*- coding: utf-8 -*-
"""
testbed of unsupervised time series anomaly detection
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import matplotlib.pyplot as plt
import os
import argparse
import getpass
import yaml
import time
import importlib as imp
import numpy as np
from testbed.utils import import_ts_data_unsupervised, get_lr
from deepod.metrics import ts_metrics, point_adjustment
import pandas as pd
from insert_outlier import insert_outlier
from sample_selection.DQNSS import DQNSS
from sample_selection.QSS import QSS
from sample_selection.ENV import ADEnv
# from deepod.utils.utility import insert_pollution, insert_pollution_seq, insert_pollution_new, split_pollution

dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'
dataset_root_DC = f'/home/{getpass.getuser()}/dataset/5-TSdata/_DCDetector/'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--output_dir", type=str, default='@records/',
                    help="the output file path")
parser.add_argument("--trainsets_dir", type=str, default='@trainsets/',
                    help="the output file path")

parser.add_argument("--dataset", type=str,
                    default='ASD',
                    help='ASD,MSL,SMAP,SMD,SWaT_cut,PUMP,DASADS,UCR_natural_fault,UCR_natural_gait,UCR_natural_heart_sbeat'
                         'UCR_natural_fault,UCR_natural_gait,UCR_natural_heart_sbeat',
                    # help='WADI,PUMP,PSM,ASD,SWaT_cut,DASADS,EP,UCR_natural_mars,UCR_natural_insect,UCR_natural_heart_vbeat2,'
                    #      'UCR_natural_heart_vbeat,UCR_natural_heart_sbeat,UCR_natural_gait,UCR_natural_fault'
                    )
parser.add_argument("--entities", type=str,
                    default='FULL',      # ['C-1', 'C-2', 'F-4']
                    help='FULL represents all the csv file in the folder, '
                         'or a list of entity names split by comma '    # ['D-14', 'D-15'], ['D-14']
                    )
parser.add_argument("--entity_combined", type=int, default=1, help='1:merge, 0: not merge')
parser.add_argument("--model", type=str, default='NCAD',
                    help="TcnED, TranAD, NCAD, NeuTraLTS, LSTMED, TimesNet, AnomalyTransformer"
                    )

parser.add_argument('--silent_header', type=bool, default=False)
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')

parser.add_argument('--seq_len', type=int, default=30)
parser.add_argument('--stride', type=int, default=1)

parser.add_argument('--sample_selection', type=int, default=0)      # 0：不划窗，1：min划窗
parser.add_argument('--insert_outlier', type=int, default=1)      # 0不插入异常，1插入异常
parser.add_argument('--rate', type=int, default=10)                # 异常数目
args = parser.parse_args()

# rate_list = [0, 0.01, 0.02, 0.1, 0.15, 0.2]
# rate_list = [0.2, 0.4, 0.6, 0.8]
# rate = 0.2

module = imp.import_module('deepod.models')
model_class = getattr(module, args.model)

path = 'testbed/configs.yaml'
with open(path) as f:
    d = yaml.safe_load(f)
    try:
        model_configs = d[args.model]
    except KeyError:
        print(f'config file does not contain default parameter settings of {args.model}')
        model_configs = {}
model_configs['seq_len'] = args.seq_len
model_configs['stride'] = args.stride


def plot(xTest, yTest, xPred, adj_score, score):
    if len(xPred) != len(xTest):
        new_xPred = [i[0][0] for i in xPred[:-1]]
        last = [i[0] for i in xPred[-1]]
        new_xPred = np.concatenate([new_xPred, last])
        xPred = new_xPred

    xTest = [i[0] for i in xTest]

    score = np.abs(xTest - xPred)

    t = np.percentile(score, (1-sum(yTest)*2.5/len(yTest))*100)
    index = np.where(score > t)[0]
    adj_yPred = np.zeros(len(score))
    adj_yPred[index] = 1

    length = 5000
    splits = np.where(yTest[1:] != yTest[:-1])[0] + 1
    is_anomaly = yTest[0] == 1
    timestamp = np.arange(len(xTest))
    pos = 0
    end = 0
    # for sp in splits:
    #     if is_anomaly:
    #         if sp < end:
    #             is_anomaly = not is_anomaly
    #             pos = sp
    #             continue
    #         left = int((length - (sp - pos)) / 2)
    #         start = max(0, pos - left)
    #         end = min(len(yTest), sp + left)
    #         xTest_plot = xTest[start:end]
    #         xPred_plot = xPred[start:end]
    #         yTest_plot = yTest[start:end]
    #         yPred_plot = adj_yPred[start:end]
    #         residual = score[start:end]
    #         ts = timestamp[start:end]
    #         plt.suptitle(args.model)
    #
    #         plt.subplot(511)
    #         plt.plot(ts, xTest_plot)
    #         plt.ylabel('xTest')
    #
    #         plt.subplot(512)
    #         plt.plot(ts, xPred_plot)
    #         plt.ylabel('xPred')
    #
    #         plt.subplot(513)
    #         plt.plot(ts, residual)
    #         plt.ylabel('residual')
    #
    #         plt.subplot(514)
    #         plt.plot(ts, yTest_plot)
    #         plt.ylabel('yTest')
    #
    #         plt.subplot(515)
    #         plt.plot(ts, yPred_plot)
    #         plt.ylabel('yPred')
    #
    #         plt.show()
    #     is_anomaly = not is_anomaly
    #     pos = sp

    plt.suptitle(args.model+'-kpi12')
    plt.subplot(511)
    plt.plot(timestamp, xTest)
    plt.ylabel('xTest')

    plt.subplot(512)
    plt.plot(timestamp, xPred)
    plt.ylabel('xPred')

    plt.subplot(513)
    plt.plot(timestamp, score)
    plt.ylabel('residual')

    plt.subplot(514)
    plt.plot(timestamp, yTest)
    plt.ylabel('yTest')

    plt.subplot(515)
    plt.plot(timestamp, adj_yPred)
    plt.ylabel('yPred')
    plt.show()


def main():
    # # setting result file/folder path
    cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f'{args.model}.{args.flag}.csv')
    # # setting loss file/folder path
    funcs = ['norm', 'myfunc-addo', 'Arxiv17', None, None, 'ICLM21', 'Arxiv22', 'myfunc', 'DQN-myfunc']
    trainsets_dir = f'{args.trainsets_dir}/{args.model}.{args.flag}/'
    os.makedirs(trainsets_dir, exist_ok=True)

    # # print header in the result file
    if not args.silent_header:
        f = open(result_file, 'a')
        print('\n---------------------------------------------------------', file=f)
        print(f'model: {args.model}, dataset: {args.dataset}, '
              f'{args.runs}runs, {cur_time}', file=f)
        for k in model_configs.keys():
            print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
        print(f'Parameters,\t [funcs], \t\t  {funcs[args.sample_selection]}', file=f)
        print(f'Note: {args.note}', file=f)
        print(f'---------------------------------------------------------', file=f)
        print(f'data, adj_auroc, std, adj_ap, std, adj_f1, std, adj_p, std, adj_r, std, time, model', file=f)
        f.close()
        print('write')
        print(args.insert_outlier, args.rate, args.sample_selection)

    dataset_name_lst = args.dataset.split(',')

    for dataset in dataset_name_lst:
        # # import data
        # if dataset in ['MSL', 'SMAP', 'SMD'] or 'UCR' in dataset:
        #     model_configs['seq_len'] = 100
        # else:
        #     model_configs['seq_len'] = 30

        data_pkg = import_ts_data_unsupervised(dataset_root, dataset_root_DC,
                                                     dataset, entities=args.entities,
                                                     combine=args.entity_combined)
        train_lst, test_lst, label_lst, name_lst = data_pkg

        entity_metric_lst = []
        entity_metric_std_lst = []
        entity_t_lst = []
        for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
            # train_data, train_labels = insert_pollution(train_data, test_data, labels, args.rate, args.seq_len)
            # train_data, train_labels, test_data, labels = insert_pollution_new(test_data, labels, args.rate)
            # train_seq_o, train_seq_l, test_data, labels = insert_pollution_seq(test_data, labels, args.rate, args.seq_len)
            # train_data, train_labels, test_data, labels = split_pollution(test_data, labels)
            # extreme, shift, trend, variance
            if args.insert_outlier:
                train_data, train_labels = insert_outlier(dataset, train_data, args.rate, 'variance')

            entries = []
            t_lst = []
            lr, epoch, a = get_lr(dataset_name, args.model, args.insert_outlier, model_configs['lr'], model_configs['epochs'])
            model_configs['lr'] = lr
            model_configs['epochs'] = epoch
            model_configs['a'] = a
            print(f'Model Configs: {model_configs}')
            for i in range(args.runs):
                print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] [{funcs[args.sample_selection]}] on Dataset [{dataset_name}]')
                print(f'\ninsert outlier [{args.insert_outlier}] with pollution rate [{args.rate}]')

                t1 = time.time()
                clf = model_class(**model_configs, random_state=83+i)
                clf.sample_selection = args.sample_selection
                if args.sample_selection != 8:
                    # clf.fit(None, None, test_data, labels, train_seq_o, train_seq_l)
                    # clf.fit(train_data, train_labels, test_data, labels)
                    clf.fit(train_data, None, test_data, labels)
                    # clf.fit(train_data)
                    # clf.fit(test_data, labels)
                    # clf.fit(train_data, labels)
                else:
                    env = ADEnv(
                        dataset=train_data,
                        y=None,
                        clf=clf
                    )
                    dqnss = DQNSS(env)
                    # dqnss = QSS(env)
                    dqnss.OD_fit(test_data, labels)

                t = time.time() - t1

                scores = clf.decision_function(test_data)
                eval_metrics = ts_metrics(labels, scores)
                adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
                # plot(test_data, labels, clf.xPred, point_adjustment(labels, scores), scores)

                # print single results
                txt = f'{dataset_name},'
                txt += ', '.join(['%.4f' % a for a in eval_metrics]) + \
                       ', pa, ' + \
                       ', '.join(['%.4f' % a for a in adj_eval_metrics])
                txt += f', model, {args.model}, time, {t:.1f}, runs, {i+1}/{args.runs}, {funcs[args.sample_selection]}'
                print(txt)

                entries.append(adj_eval_metrics)
                t_lst.append(t)

                if not args.silent_header:
                    trainsets_df = pd.DataFrame.from_dict(clf.trainsets, orient='index').transpose()
                    trainsets_df.to_csv(trainsets_dir + dataset_name + '_' + funcs[args.sample_selection] + str(args.rate*args.insert_outlier) + str(i)+'.csv', index=False)
                    if len(clf.result_detail) != 0:
                        df_result = pd.DataFrame(clf.result_detail, columns=['auc', 'pr', 'f1', 'adjauc', 'adjpr', 'adjf1'])
                        df_result.to_csv(os.path.join(args.output_dir, f'{args.model}.{dataset_name}.{funcs[args.sample_selection]}.{args.rate*args.insert_outlier}.{i}.csv'))
                    if args.sample_selection == 2:
                        clf.Arxiv17['std_avg'] = (clf.Arxiv17['avg']-np.min(clf.Arxiv17['avg']))/(np.max(clf.Arxiv17['avg'])-np.min(clf.Arxiv17['avg']))
                        clf.Arxiv17['std_std'] = (clf.Arxiv17['std']-np.min(clf.Arxiv17['std']))/(np.max(clf.Arxiv17['std'])-np.min(clf.Arxiv17['std']))
                        Arxiv17_df = pd.DataFrame.from_dict(clf.Arxiv17, orient='index').transpose()
                        Arxiv17_df.to_csv(trainsets_dir + dataset_name + '_' + funcs[args.sample_selection] + str(args.rate*args.insert_outlier) + str(i)+'.csv', index=False)

            avg_entry = np.average(np.array(entries), axis=0)
            std_entry = np.std(np.array(entries), axis=0)
            avg_t = np.average(t_lst)

            entity_metric_lst.append(avg_entry)
            entity_metric_std_lst.append(std_entry)
            entity_t_lst.append(avg_t)

            if 'UCR' not in dataset_name or args.entity_combined == 1:
                txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                      '%.4f, %.4f, %.4f, %.4f, %.1f, %s, %f ' % \
                      (dataset_name,
                       avg_entry[0], std_entry[0], avg_entry[1], std_entry[1],
                       avg_entry[2], std_entry[2], avg_entry[3], std_entry[3],
                       avg_entry[4], std_entry[4],
                       avg_t, args.model+'-'+funcs[args.sample_selection]+str(args.insert_outlier*args.rate), model_configs['lr'])
                print(txt)

                if not args.silent_header:
                    f = open(result_file, 'a')
                    print(txt, file=f)
                    f.close()

        if 'UCR' in dataset or args.entity_combined == 0:
            entity_avg_mean = np.average(np.array(entity_metric_lst), axis=0)
            entity_std_mean = np.average(np.array(entity_metric_std_lst), axis=0)

            txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                  '%.4f, %.4f, %.4f, %.4f, %.1f, %s, %f ' % \
                  (dataset,
                   entity_avg_mean[0], entity_std_mean[0], entity_avg_mean[1], entity_std_mean[1],
                   entity_avg_mean[2], entity_std_mean[2], entity_avg_mean[3], entity_std_mean[3],
                   entity_avg_mean[4], entity_std_mean[4],
                   np.sum(np.array(entity_t_lst)),
                   args.model + '-' + funcs[args.sample_selection] + str(args.insert_outlier * args.rate),
                   model_configs['lr'])
            print(txt)

            if not args.silent_header:
                f = open(result_file, 'a')
                print(txt, file=f)
                f.close()


def count_datasets(dname, entities, combine):

    dataset_name_lst = dname.split(',')
    for dataset in dataset_name_lst:
        # # import data
        data_pkg = import_ts_data_unsupervised(dataset_root,
                                               dataset, entities=entities,
                                               combine=combine)
        train_lst, test_lst, label_lst, name_lst = data_pkg

        Ntrain = []
        Ntest = []
        Otest = []
        Feature = []
        for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
            Ntrain.append(len(train_data))
            Ntest.append(len(test_data))
            Otest.append(sum(labels))
            Feature.append(len(train_data[0]))
            if combine == 1:
                print('%s,%d,%d,%d,%d' % (dataset, len(train_data), len(test_data), sum(labels), len(train_data[0])))
        if combine == 0:
            print('%s,%d,%d,%d,%d' % (dataset, np.average(Ntrain), np.average(Ntest), np.average(Otest), np.average(Feature)))


def print_dataset():
    print("dataset_name,|N-train|,|N-test|,|O-test|,|Feature|")
    dname = 'ASD,DASADS,PUMP,SMD,MSL,SMAP,SWaT_cut'
    dname_mean = 'UCR_natural_heart_vbeat,UCR_natural_heart_vbeat2,UCR_natural_fault,UCR_natural_gait,UCR_natural_heart_sbeat,UCR_natural_insect,UCR_natural_mars'
    count_datasets(dname, 'FULL', 1)
    count_datasets(dname_mean, 'FULL', 0)


def print_Nused():
    import glob
    func = 'TcnED'

    # dname = 'ASD,DASADS'              # 'NeuTraLTS'
    # dname = 'ASD,MSL,SMAP,SMD'      # 'TcnED'
    # dname = 'ASD,DASADS'  #   'NCAD'
    # dname = 'ASD,DASADS,SMD,MSL,SMAP,SWaT_cut'
    dname = 'DASADS'
    dataset_name_lst = dname.split(',')
    for d in dataset_name_lst:
        length = []
        for i in range(5):
            # path = '/home/xuhz/zry/DeepOD-new/@trainsets/%s./%s_combined_myfunc0.0%d.csv' % (func, d, i)
            path = '/home/xuhz/zry/DeepOD-new/@trainsets/%s./%s_combined_myfunc0%d.csv' % (func, d, i)
            # /home/xuhz/zry/DeepOD-new/@trainsets/NCAD./SMD_combined_myfunc0.00.csv
            df = pd.read_csv(path)
            column = df.columns[-1]
            df = df[column].values
            length.append(len(df))
        print('%s,%s,%d' % (func, d, np.average(length)))

    # dname = 'PUMP'      # NCAD
    # dname = 'PUMP'      # NeuTraLTS
    dname = 'SWaT_cut,PUMP'
    dataset_name_lst = dname.split(',')
    for d in dataset_name_lst:
        length = []
        for i in range(5):
            # path = '/home/xuhz/zry/DeepOD-new/@trainsets/%s./%s_myfunc0.0%d.csv' % (func, d, i)
            path = '/home/xuhz/zry/DeepOD-new/@trainsets/%s./%s_myfunc0%d.csv' % (func, d, i)
            # /home/xuhz/zry/DeepOD-new/@trainsets/NCAD./SMD_combined_myfunc0.00.csv
            df = pd.read_csv(path)
            column = df.columns[-1]
            df = df[column].values
            length.append(len(df))
        print('%s,%s,%d' % (func, d, np.average(length)))

    # dname = 'UCR_natural_fault,UCR_natural_gait,UCR_natural_heart_sbeat'
    # dataset_name_lst = dname.split(',')
    # for data in dataset_name_lst:
    #     machine_lst = os.listdir(dataset_root + data + '/')
    #     length = []
    #     for m in machine_lst:
    #         for i in range(5):
    #             # path = '/home/xuhz/zry/DeepOD-new/@trainsets/%s./%s-%s_myfunc0.0%d.csv' % (func, data, m, i)
    #             path = '/home/xuhz/zry/DeepOD-new/@trainsets/%s./%s-%s_myfunc0%d.csv' % (func, data, m, i)
    #             # /home/xuhz/zry/DeepOD-new/@trainsets/NCAD./SMD_combined_myfunc0.00.csv
    #             df = pd.read_csv(path)['dis10'].values
    #             length.append(len(df))
    #     print('%s,%s,%d' % (func, data, np.average(length)))


if __name__ == '__main__':
    # print_Nused()
    # print_dataset()
    main()