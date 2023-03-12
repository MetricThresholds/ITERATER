#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ3_TAL_10_percent_spv_apv_all_percents.py
Date : 2022/11/20 19:44

将  RQ3_TAL_spv_all_percents.py  与   RQ3_TAL_apv_all_percents.py  的结果执行结果中按10%，20%，...,90%百分比得出的阈值性能
按HALKP方法格式整理好，便于与TAL方法做比较。
"""


def ten_percent_performance(df_name, last_iteration, percent):
    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    F_measure_5_percent = \
        df_name[df_name['iteration'] == last_iteration + '_threshold'].loc[:, 'f1_' + str(round(percent, 1))].values[0]

    g_mean_5_percent = \
        df_name[df_name['iteration'] == last_iteration + '_threshold'].loc[:, 'gm_' + str(round(percent, 1))].values[0]

    Balance_5_percent = \
        df_name[df_name['iteration'] == last_iteration + '_threshold'].loc[:, 'bpp_' + str(round(percent, 1))].values[0]

    return F_measure_5_percent, g_mean_5_percent, Balance_5_percent


def collect_data_from_active_learning(working_dir, result_dir):
    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    percent_dir = result_dir + 'TAL_90_percent_al_10_percent/'

    if not os.path.exists(percent_dir):
        os.mkdir(percent_dir)

    percent_lists = ['5_percent', '10_percent', '15_percent', '20_percent']

    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    # store the data
    df_als = pd.DataFrame(
        columns=['Project', 'Prior', 'Current', 'metrics', 'threshold_5%', 'threshold_10%', 'threshold_15%',
                 'threshold_20%', 'F-measure_5_percent', 'g-mean_5_percent', 'Balance_5_percent',
                 'F-measure_10_percent', 'g-mean_10_percent', 'Balance_10_percent', 'F-measure_15_percent',
                 'g-mean_15_percent', 'Balance_15_percent', 'F-measure_20_percent', 'g-mean_20_percent',
                 'Balance_20_percent'])
    # 按HALKP的格式整理各度量的数据，以方便比较
    df_halkp = pd.read_csv(working_dir + 'XuResults/' + 'HALKP_5_10_15_20_percent_spv_apv.csv')
    print(df_halkp.head())

    for percent in range(9):

        print(percent, percent * 0.1)

        for i in range(len(df_halkp)):
            cvdp_halkp = str(df_halkp.loc[i, 'cvdp'])
            Project_halkp = str(df_halkp.loc[i, 'Project'])
            Current_halkp = str(df_halkp.loc[i, 'Current'])
            print(cvdp_halkp, Project_halkp, Current_halkp)

            # store the data
            if os.path.exists(percent_dir + 'al_' + str(round((percent + 1) * 0.1, 1)) + '_percent.csv'):
                df_als_metric = pd.read_csv(
                    percent_dir + 'al_' + str(round((percent + 1) * 0.1, 1)) + '_percent.csv')
            else:
                df_als_metric = pd.DataFrame(
                    columns=['cvdp', 'Project', 'Prior', 'Current',
                             'F-measure_5_percent', 'g-mean_5_percent', 'Balance_5_percent',
                             'F-measure_10_percent', 'g-mean_10_percent', 'Balance_10_percent',
                             'F-measure_15_percent', 'g-mean_15_percent', 'Balance_15_percent',
                             'F-measure_20_percent', 'g-mean_20_percent', 'Balance_20_percent'])
            Prior = ''
            F_measure_5_percent, g_mean_5_percent, Balance_5_percent = 0, 0, 0
            F_measure_10_percent, g_mean_10_percent, Balance_10_percent = 0, 0, 0
            F_measure_15_percent, g_mean_15_percent, Balance_15_percent = 0, 0, 0
            F_measure_20_percent, g_mean_20_percent, Balance_20_percent = 0, 0, 0

            for percent_list in percent_lists:

                name_path = result_dir + cvdp_halkp + '_' + percent_list + '/TAL_90_percent' + \
                            '/activeLearningPerformance_' + Project_halkp + '-' + Current_halkp + '.csv'
                print(name_path)
                if not os.path.exists(name_path):
                    continue

                df_name = pd.read_csv(name_path)
                last_iteration = df_name.iteration.values[-1].split('_')[0]
                Prior = df_name.last_version.values[-1].replace(Project_halkp + '-', '').replace('.csv', '')
                # Current = df_name.current_version.values[-1].split('-')[-1]

                if percent_list == '5_percent':
                    F_measure_5_percent, g_mean_5_percent, Balance_5_percent = \
                        ten_percent_performance(df_name, last_iteration, (percent + 1) * 0.1)

                if percent_list == '10_percent':
                    F_measure_10_percent, g_mean_10_percent, Balance_10_percent = \
                        ten_percent_performance(df_name, last_iteration, (percent + 1) * 0.1)

                if percent_list == '15_percent':
                    F_measure_15_percent, g_mean_15_percent, Balance_15_percent = \
                        ten_percent_performance(df_name, last_iteration, (percent + 1) * 0.1)

                if percent_list == '20_percent':
                    F_measure_20_percent, g_mean_20_percent, Balance_20_percent = \
                        ten_percent_performance(df_name, last_iteration, (percent + 1) * 0.1)

            df_als_metric = df_als_metric.append(
                {'cvdp': cvdp_halkp, 'Project': Project_halkp, 'Prior': Prior, 'Current': Current_halkp,
                 'F-measure_5_percent': F_measure_5_percent, 'g-mean_5_percent': g_mean_5_percent,
                 'Balance_5_percent': Balance_5_percent,
                 'F-measure_10_percent': F_measure_10_percent, 'g-mean_10_percent': g_mean_10_percent,
                 'Balance_10_percent': Balance_10_percent,
                 'F-measure_15_percent': F_measure_15_percent, 'g-mean_15_percent': g_mean_15_percent,
                 'Balance_15_percent': Balance_15_percent,
                 'F-measure_20_percent': F_measure_20_percent, 'g-mean_20_percent': g_mean_20_percent,
                 'Balance_20_percent': Balance_20_percent}, ignore_index=True)

            df_als_metric.to_csv(
                percent_dir + '/al_' + str(round((percent + 1) * 0.1, 1)) + '_percent.csv', index=False)

            # break


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

    work_Directory = "F:/talcvdp/"
    result_Directory = "F:/talcvdp/active_learning_middle_prior_pseudo_all_percents/"
    # result_Directory = "F:/talcvdp/active_learning_middle_prior_pseudo_all_percents/spv_5_percent/TAL_10_percent/"
    # result_Directory = "F:/talcvdp/active_learning_middle_prior_pseudo/"

    # 用于存储主动学习后，剩余未被检测模块，用于spv,apv,sfv,manualdown,manulaup,等基线方法的比较。
    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    collect_data_from_active_learning(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
