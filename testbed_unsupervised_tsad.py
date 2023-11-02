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

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--output_dir", type=str, default='@records/',
                    help="the output file path")
parser.add_argument("--trainsets_dir", type=str, default='@trainsets/',
                    help="the output file path")

parser.add_argument("--dataset", type=str,
                    default='UCR_natural_heart_vbeat,UCR_natural_heart_vbeat2,SMD,MSL,SMAP,SWaT_cut',
                    help='ASD,DASADS,PUMP,UCR_natural_heart_vbeat,UCR_natural_heart_vbeat2,SMD,MSL,SMAP,SWaT_cut',
                    # help='WADI,PUMP,PSM,ASD,SWaT_cut,DASADS,EP,UCR_natural_mars,UCR_natural_insect,UCR_natural_heart_vbeat2,'
                    #      'UCR_natural_heart_vbeat,UCR_natural_heart_sbeat,UCR_natural_gait,UCR_natural_fault'
                    )
parser.add_argument("--entities", type=str,
                    default='FULL',
                    help='FULL represents all the csv file in the folder, '
                         'or a list of entity names split by comma '    # ['D-14', 'D-15'], ['D-14']
                    )
parser.add_argument("--entity_combined", type=int, default=1, help='1:merge, 0: not merge')
parser.add_argument("--model", type=str, default='NCAD',
                    help="TcnED, TranAD, NCAD, NeuTraLTS, LSTMED, TimesNet, AnomalyTransformer, DCdetector"
                    )

parser.add_argument('--silent_header', type=bool, default=False)
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')

parser.add_argument('--seq_len', type=int, default=30)
parser.add_argument('--stride', type=int, default=1)

parser.add_argument('--sample_selection', type=int, default=7)      # 0：不划窗，1：min划窗
parser.add_argument('--insert_outlier', type=float, default=0)      # 0不插入异常，1插入异常
parser.add_argument('--rate', type=int, default=20)                # 异常数目
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
        data_pkg = import_ts_data_unsupervised(dataset_root,
                                                     dataset, entities=args.entities,
                                                     combine=args.entity_combined)
        train_lst, test_lst, label_lst, name_lst = data_pkg

        entity_metric_lst = []
        entity_metric_std_lst = []
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
            lr = get_lr(dataset_name, args.model, model_configs['lr'])
            model_configs['lr'] = lr
            print(f'Model Configs: {model_configs}')
            for i in range(args.runs):
                print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] [{funcs[args.sample_selection]}] on Dataset [{dataset_name}]')
                print(f'\ninsert outlier [{args.insert_outlier}] with pollution rate [{args.rate}]')

                t1 = time.time()
                clf = model_class(**model_configs, random_state=42+i)
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

            entity_metric_lst.append(avg_entry)
            entity_metric_std_lst.append(std_entry)

            txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                  '%.4f, %.4f, %.4f, %.4f, %.1f, %s ' % \
                  (dataset_name,
                   avg_entry[0], std_entry[0], avg_entry[1], std_entry[1],
                   avg_entry[2], std_entry[2], avg_entry[3], std_entry[3],
                   avg_entry[4], std_entry[4],
                   np.average(t_lst), args.model+'-'+funcs[args.sample_selection]+str(args.insert_outlier*args.rate))
            print(txt)

            if not args.silent_header:
                f = open(result_file, 'a')
                print(txt, file=f)
                f.close()


if __name__ == '__main__':
    main()
    # for i in [0, 5, 10, 15, 20]:        # 0, 5, 6, 7
    #     print(i)
    #     args.silent_header = False
    #     args.insert_outlier = 1
    #     args.sample_selection = 7
    #     args.runs = 5
    #     args.rate = i
    #     main()

    # for i in [7]:
    #     print(i)
    #     args.insert_outlier = 0
    #     args.sample_selection = i
    #     args.runs = 5
    #     args.rate = 0
    #     main()

    # for rate in [0, 0.2, 0.4, 0.6, 0.8]:
    #     print(rate)
    #     args.rate = rate
    #     args.runs = 1
    #     args.sample_selection=0
    #     main()

    # args.rate = 0
    # args.runs = 1
    # args.sample_selection=7
    # args.model = 'TcnED'
    # main()

    # args.rate = 0.1
    # args.sample_selection = 1
    # args.runs = 0
    # main()

    # for model in ['TcnED', 'TimesNet', 'TranAD', 'AnomalyTransformer']:
    #     args.runs = 5
    #     args.model = model
    #     args.sample_selection = 7
    #     main()