# -*- coding: utf-8 -*-
"""
testbed of unsupervised time series anomaly detection
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

import os
import argparse
import getpass
import yaml
import time
import importlib as imp
import numpy as np
from testbed.utils import import_ts_data_unsupervised
from deepod.metrics import ts_metrics, point_adjustment
import pandas as pd

dataset_root = f'/home/{getpass.getuser()}/dataset/5-TSdata/_processed_data/'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--output_dir", type=str, default='@records/',
                    help="the output file path")
parser.add_argument("--loss_dir", type=str, default='@losses/',
                    help="the output file path")
parser.add_argument("--dataset", type=str,
                    default='ASD,SMAP,MSL',
                    help='ASD,SMAP,MSL,SWaT_cut'
                    )
parser.add_argument("--entities", type=str,
                    default='FULL',
                    help='FULL represents all the csv file in the folder, '
                         'or a list of entity names split by comma '    # ['D-14', 'D-15'], ['D-14']
                    )
parser.add_argument("--entity_combined", type=int, default=1, help='1:merge, 0: not merge')
parser.add_argument("--model", type=str, default='TcnED', help="TimesNet")

parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')

parser.add_argument('--seq_len', type=int, default=30)
parser.add_argument('--stride', type=int, default=1)

parser.add_argument('--sample_selection', type=int, default=3)      # 0：不划窗，1：min划窗

args = parser.parse_args()

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

print(f'Model Configs: {model_configs}')


# # setting result file/folder path
cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
os.makedirs(args.output_dir, exist_ok=True)
result_file = os.path.join(args.output_dir, f'{args.model}.{args.flag}.csv')
# # setting loss file/folder path
funcs = ['norm', 'min', 'distribution', 'imp_param']
loss_dir = f'{args.loss_dir}/{args.model}.{args.flag}/'
os.makedirs(loss_dir, exist_ok=True)


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

        entries = []
        t_lst = []
        for i in range(args.runs):
            start_time = time.time()
            print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

            t1 = time.time()
            clf = model_class(**model_configs, random_state=42+i)
            clf.sample_selection = args.sample_selection
            clf.fit(train_data, None, test_data, labels)
            t = time.time() - t1

            scores = clf.decision_function(test_data)
            eval_metrics = ts_metrics(labels, scores)
            adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))

            # print single results
            txt = f'{dataset_name},'
            txt += ', '.join(['%.4f' % a for a in eval_metrics]) + \
                   ', pa, ' + \
                   ', '.join(['%.4f' % a for a in adj_eval_metrics])
            txt += f', model, {args.model}, time, {t:.1f}, runs, {i+1}/{args.runs}'
            print(txt)

            entries.append(adj_eval_metrics)
            t_lst.append(t)

            os.path.join(args.output_dir, f'{args.model}.{args.flag}.csv')
            loss_df = pd.DataFrame(clf.loss_by_epoch)
            loss_df.to_csv(loss_dir + dataset_name + '_' + funcs[args.sample_selection] + str(i)+'.csv', index=False)

        avg_entry = np.average(np.array(entries), axis=0)
        std_entry = np.std(np.array(entries), axis=0)

        entity_metric_lst.append(avg_entry)
        entity_metric_std_lst.append(std_entry)

        f = open(result_file, 'a')
        txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
              '%.4f, %.4f, %.4f, %.4f, %.1f, %s ' % \
              (dataset_name,
               avg_entry[0], std_entry[0], avg_entry[1], std_entry[1],
               avg_entry[2], std_entry[2], avg_entry[3], std_entry[3],
               avg_entry[4], std_entry[4],
               np.average(t_lst), args.model+'-'+str(args.sample_selection))
        print(txt)
        print(txt, file=f)
        f.close()
