#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Project : ActiveLearing
@File : RQ2_EASC_on_al_test_apv.py
@Time : 2023/3/18 0:28

应用easc-npm方法用前一版本的真实标签数据训练模型，然后在当前版本与ITERATER相同测试集的数据上预测性能，以方便比较
"""


def prediction_performance(bugBinary, predictBinary):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
    c_matrix = confusion_matrix(bugBinary, predictBinary, labels=[0, 1])
    tn, fp, fn, tp = c_matrix.ravel()

    if (tn + fp) == 0:
        tnr_value = 0
    else:
        tnr_value = tn / (tn + fp)

    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    recall_value = recall_score(bugBinary, predictBinary, labels=[0, 1])
    precision_value = precision_score(bugBinary, predictBinary, labels=[0, 1])
    f1_value = f1_score(bugBinary, predictBinary, labels=[0, 1])
    gm_value = (recall_value * tnr_value) ** 0.5
    pdr = recall_value
    pfr = fpr  # fp / (fp + tn)
    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = (fp + fn) / (tp + fp + fn + tn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc


def random_forest_on_tal_test(working_dir, result_dir):

    from sklearn.naive_bayes import GaussianNB

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all rows and columns of a matrix
    np.set_printoptions(threshold=np.sys.maxsize, linewidth=np.sys.maxsize)

    dir_spv_nb = result_dir + 'NB/apv_20_percent/'
    dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/apv_20_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/apv_15_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/apv_10_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/apv_5_percent/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists(result_dir + 'NB/'):
        os.mkdir(result_dir + 'NB/')

    if not os.path.exists(dir_spv_nb):
        os.mkdir(dir_spv_nb)

    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    with open(working_dir + 'List.txt') as l_projects:
        projects_t = l_projects.readlines()

    for project_t in projects_t:

        project = project_t.replace("\n", "")

        # read each version of a project in order
        versions_sorted = []
        with open(working_dir + 'List_versions.txt') as l_versions:
            versions_t = l_versions.readlines()

        for version_t in versions_t:
            if version_t.split('-')[0] == project:
                versions_sorted.append(version_t.replace('\n', ''))
        print(versions_sorted)

        # df_name_previous: previous version of current version. df_previous_all: all previous versions of current one.
        df_name_previous = pd.DataFrame()
        df_previous_all = pd.DataFrame()
        name_previous = ''
        name_previous_all = ''

        for name in versions_sorted:
            # store the results of active_learning_semisupervised：'f1_value', 'gm_value', 'bpp_value'
            df_nb = pd.DataFrame(
                columns=['Sample_size', 'project', 'last_version', 'current_version', 'recall_value', 'precision_value',
                         'f1_value', 'gm_value', 'bpp_value', 'accuracy', 'error_rate', 'mcc'], dtype=object)

            # df_name for spv
            df_name = pd.read_csv(working_dir + project + '/' + name)

            # 与for循环最后df_name_previous = df_name互应，若第一个赋值给上一个，进入下一循环，否则在for循环最后一句把当前赋值给上一个
            if name == versions_sorted[0]:
                df_name_previous = df_name
                name_previous = name
                continue

            # bugBinary表示bug的二进制形式, 'iter':表示该行(模块)是否被选中，初始为零，表示都没选中,反之选中
            df_name_previous["bugBinary"] = df_name_previous.bug.apply(lambda x: 1 if x > 0 else 0)
            df_name_previous['iter'] = 1

            # APV scenario： 合并当前版本与前一版本
            df_previous_all = df_previous_all.append(df_name_previous).reset_index(drop=True)
            if len(name_previous_all) == 0:
                name_previous_all = name_previous_all + name_previous[:-4]
            else:
                name_previous_all = name_previous_all + '_' + name_previous[:-4]

            # 仅当当前版本为最后版本时才开始迭代主动学习选择模块
            if name == versions_sorted[len(versions_sorted) - 1]:

                df_name_test = pd.read_csv(dir_test + 'testing_data_' + name)

                print(len(df_previous_all))

                # Gaussian Naive Bayes
                gnb = GaussianNB()

                gnb.fit(df_previous_all.loc[:, metrics_20], df_previous_all.loc[:, 'bugBinary'])

                predict_nb_proba = gnb.predict_proba(df_name_test.loc[:, metrics_20])

                predict_nb = []
                for i in range(len(predict_nb_proba)):
                    predict_nb.append(predict_nb_proba[i][1])

                df_name_test['predict_nb'] = np.array(predict_nb)
                df_name_test['predict_nb_EPM'] = df_name_test.apply(lambda x: x['predict_nb'] / x['loc'], axis=1)
                df_name_test['predict_nb_NPM'] = df_name_test.apply(lambda x: x['predict_nb'] * x['loc'], axis=1)

                df_testing_defective = pd.DataFrame()
                df_testing_non_defective = pd.DataFrame()
                for j in range(len(df_name_test)):
                    if df_name_test.loc[j, 'predict_nb'] > 0.5:
                        df_testing_defective = df_testing_defective.append(df_name_test.loc[j, :])
                    else:
                        df_testing_non_defective = df_testing_non_defective.append(df_name_test.loc[j, :])

                # 若没有预测为有预测的模块，则df_testing_defective为空df_testing_defective
                if len(df_testing_defective) > 0:
                    df_testing_defective.sort_values(["predict_nb_NPM"], inplace=True, ascending=False)
                if len(df_testing_non_defective) > 0:
                    df_testing_non_defective.sort_values(["predict_nb_NPM"], inplace=True, ascending=False)

                df_testing_defective = df_testing_defective.append(df_testing_non_defective)

                df_testing_defective['easc'] = range(1, len(df_testing_defective) + 1)
                threshold = df_testing_defective.easc.quantile(0.2)
                df_testing_defective['predict_nb_binary'] = df_testing_defective.easc.apply(lambda x: 1 if x <= threshold else 0)

                print(df_name_test.columns.values)
                recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc = \
                    prediction_performance(df_testing_defective['bugBinary'], df_testing_defective['predict_nb_binary'])

                print(recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc)
                df_nb = df_nb.append(
                    {'Sample_size': len(df_name_test), 'project': name.split('-')[0],
                     'last_version': name_previous[:-4],
                     'current_version': name[:-4], 'recall_value': recall_value, 'precision_value': precision_value,
                     'f1_value': f1_value, 'gm_value': gm_value, 'bpp_value': bpp_value, 'accuracy': accuracy,
                     'error_rate': error_rate, 'mcc': mcc}, ignore_index=True)
                df_nb.to_csv(dir_spv_nb + 'NaiveBayes_npm_' + name, index=False)

            # 当前版本作为下一版本的前一版本
            df_name_previous = df_name.copy()
            name_previous = name
            print('********************this is an end of a version ' + name + '*******************************')


if __name__ == '__main__':
    import os
    import sys
    import csv
    import math
    import time
    import random
    import shutil
    from datetime import datetime
    import pandas as pd
    import numpy as np

    s_time = time.time()

    work_Directory = "F:/talcvdp/Xudata/MORPH_projects/"
    result_Directory = "F:/talcvdp/easc_on_TAL_test/"
    print(result_Directory[:-26])

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    random_forest_on_tal_test(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")