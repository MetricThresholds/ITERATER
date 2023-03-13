#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ3_boxplot_for_TAL_all_percents_4_threholds_togather.py
Date : 2022/11/20 20:35

RQ3_boxplot_for_TAL_all_percents。py相似，但将5%，10%，15%，20%画在一张图上。
画出5%，10%，15%，20%四种比例上，F1,GM,BPP三个箱线图

"""


def store_plot(plot_dir, df_F_measure_5_percent, name):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    plt.rcParams['savefig.dpi'] = 900  # 图片像素
    plt.rcParams['figure.figsize'] = (4.0, 4.0)
    # plt.rcParams['figure.figsize'] = (6.0, 6.0)
    # plt.rcParams['figure.figsize'] = (10.0, 10.0)
    # plt.rcParams['figure.figsize'] = (20.0, 6.0)
    df_F_measure_5_percent.plot.box(showmeans=True, title=name)
    # df_F_measure_5_percent.plot.box(showmeans=True, title=name, fontsize=18)
    # df_F_measure_5_percent.plot.box(showmeans=True)
    # df_universal.plot.box(title="Box Plot of GM values for OO Metrics with threshold")
    plt.grid(linestyle="--", alpha=0.3)
    # plt.ylim(0.2, 0.8)
    # plt.xticks(rotation=0, fontsize=9.0)
    plt.xticks(rotation=0, fontsize=15.0)
    # plt.xticks(rotation=30, fontsize=15.0)
    plt.gcf().subplots_adjust(bottom=0.14) #这是增加下面标签的显示宽度，20210807找了下午5个小时
    # plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
    y_major_locator = MultipleLocator(0.05) #显示Y轴的刻度
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig(plot_dir + name + '.png')
    # plt.savefig(plot_dir + 'F_measure_5_percent' + '.png')
    plt.close()


def f1_gm_bpp_plot(working_dir, plot_dir):
    import os
    import csv
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    plt.rcParams['savefig.dpi'] = 900  # 图片像素
    plt.rcParams['figure.figsize'] = (8.0, 4.0)

    os.chdir(working_dir)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # metrics_26 = ['LCOM1', 'LCOM2', 'LCOM3', 'LCOM5', 'NewLCOM5', 'ICH', 'CAMC', 'NHD', 'SNHD', 'OCAIC', 'OCMIC',
    #               'OMMIC', 'CBO', 'DAC', 'ICP', 'MPC', 'NIHICP', 'RFC', 'NMA', 'NA', 'NAIMP', 'NM', 'NMIMP', 'NumPara',
    #               'SLOC', 'Stmts']
    percent_11 = ['Max.', 'Median', 'Min.']
    # percent_11 = ['1%', '50%', '100%']
    # percent_11 = ['10%', '50%', '90%']
    # percent_11 = ['10%', '20%', '25%', '30%', '40%', '50%', '60%', '70%', '75%', '80%', '90%']

    threshold_4 = ['5%', '10%', '15%', '20%']

    df_F_measure_max = pd.DataFrame(columns=threshold_4)
    df_g_mean_max = pd.DataFrame(columns=threshold_4)
    df_Balance_max = pd.DataFrame(columns=threshold_4)

    df_F_measure_min = pd.DataFrame(columns=threshold_4)
    df_g_mean_min = pd.DataFrame(columns=threshold_4)
    df_Balance_min = pd.DataFrame(columns=threshold_4)

    # for threshold in threshold_4:


    df_F_measure_5_percent = pd.DataFrame(columns=percent_11)
    df_F_measure_10_percent = pd.DataFrame(columns=percent_11)
    df_F_measure_15_percent = pd.DataFrame(columns=percent_11)
    df_F_measure_20_percent = pd.DataFrame(columns=percent_11)
    df_g_mean_5_percent = pd.DataFrame(columns=percent_11)
    df_g_mean_10_percent = pd.DataFrame(columns=percent_11)
    df_g_mean_15_percent = pd.DataFrame(columns=percent_11)
    df_g_mean_20_percent = pd.DataFrame(columns=percent_11)
    df_Balance_5_percent = pd.DataFrame(columns=percent_11)
    df_Balance_10_percent = pd.DataFrame(columns=percent_11)
    df_Balance_15_percent = pd.DataFrame(columns=percent_11)
    df_Balance_20_percent = pd.DataFrame(columns=percent_11)

    # df_universal = pd.DataFrame(columns=metrics_26)

    # for root, dirs, files in os.walk(working_dir):

    for percent in percent_11:

        # if name.split('_')[0] not in metrics_26:
        #     continue
        # print("The release is ", name.split('_')[0])
        # df = pd.read_csv(working_dir + name, keep_default_na=False, na_values=[""])
        # print('TAL_' + percent[:-1] + '_percent_al_10_percent/al_0.5_percent.csv')
        # df = pd.read_csv(working_dir + 'TAL_' + percent[:-1] + '_percent_al_10_percent/al_0.5_percent.csv',
        #                  keep_default_na=False, na_values=[""])
        if percent == 'Max.':
            df = pd.read_csv(working_dir + 'TAL_1_percent_al_10_percent/al_0.5_percent.csv', keep_default_na=False,
                             na_values=[""])

        if percent == 'Median':
            df = pd.read_csv(working_dir + 'TAL_50_percent_al_10_percent/al_0.5_percent.csv', keep_default_na=False,
                             na_values=[""])

        if percent == 'Min.':
            df = pd.read_csv(working_dir + 'TAL_100_percent_al_10_percent/al_0.5_percent.csv', keep_default_na=False,
                             na_values=[""])

        print(df.head())
        print(df_F_measure_5_percent)
        for i in range(len(df)):
            df_F_measure_5_percent.loc[i, percent] = df.loc[i, 'F-measure_5_percent']
            df_F_measure_10_percent.loc[i, percent] = df.loc[i, 'F-measure_10_percent']
            df_F_measure_15_percent.loc[i, percent] = df.loc[i, 'F-measure_15_percent']
            df_F_measure_20_percent.loc[i, percent] = df.loc[i, 'F-measure_20_percent']
            df_g_mean_5_percent.loc[i, percent] = df.loc[i, 'g-mean_5_percent']
            df_g_mean_10_percent.loc[i, percent] = df.loc[i, 'g-mean_10_percent']
            df_g_mean_15_percent.loc[i, percent] = df.loc[i, 'g-mean_15_percent']
            df_g_mean_20_percent.loc[i, percent] = df.loc[i, 'g-mean_20_percent']
            df_Balance_5_percent.loc[i, percent] = df.loc[i, 'Balance_5_percent']
            df_Balance_10_percent.loc[i, percent] = df.loc[i, 'Balance_10_percent']
            df_Balance_15_percent.loc[i, percent] = df.loc[i, 'Balance_15_percent']
            df_Balance_20_percent.loc[i, percent] = df.loc[i, 'Balance_20_percent']

            if percent == 'Max.':
                df_F_measure_max.loc[i, '5%'] = df.loc[i, 'F-measure_5_percent']
                df_F_measure_max.loc[i, '10%'] = df.loc[i, 'F-measure_10_percent']
                df_F_measure_max.loc[i, '15%'] = df.loc[i, 'F-measure_15_percent']
                df_F_measure_max.loc[i, '20%'] = df.loc[i, 'F-measure_20_percent']
                df_g_mean_max.loc[i, '5%'] = df.loc[i, 'g-mean_5_percent']
                df_g_mean_max.loc[i, '10%'] = df.loc[i, 'g-mean_10_percent']
                df_g_mean_max.loc[i, '15%'] = df.loc[i, 'g-mean_15_percent']
                df_g_mean_max.loc[i, '20%'] = df.loc[i, 'g-mean_20_percent']
                df_Balance_max.loc[i, '5%'] = df.loc[i, 'Balance_5_percent']
                df_Balance_max.loc[i, '10%'] = df.loc[i, 'Balance_10_percent']
                df_Balance_max.loc[i, '15%'] = df.loc[i, 'Balance_15_percent']
                df_Balance_max.loc[i, '20%'] = df.loc[i, 'Balance_20_percent']

            if percent == 'Min.':
                df_F_measure_min.loc[i, '5%'] = df.loc[i, 'F-measure_5_percent']
                df_F_measure_min.loc[i, '10%'] = df.loc[i, 'F-measure_10_percent']
                df_F_measure_min.loc[i, '15%'] = df.loc[i, 'F-measure_15_percent']
                df_F_measure_min.loc[i, '20%'] = df.loc[i, 'F-measure_20_percent']
                df_g_mean_min.loc[i, '5%'] = df.loc[i, 'g-mean_5_percent']
                df_g_mean_min.loc[i, '10%'] = df.loc[i, 'g-mean_10_percent']
                df_g_mean_min.loc[i, '15%'] = df.loc[i, 'g-mean_15_percent']
                df_g_mean_min.loc[i, '20%'] = df.loc[i, 'g-mean_20_percent']
                df_Balance_min.loc[i, '5%'] = df.loc[i, 'Balance_5_percent']
                df_Balance_min.loc[i, '10%'] = df.loc[i, 'Balance_10_percent']
                df_Balance_min.loc[i, '15%'] = df.loc[i, 'Balance_15_percent']
                df_Balance_min.loc[i, '20%'] = df.loc[i, 'Balance_20_percent']


        print(df_F_measure_5_percent)

        # break

    store_plot(plot_dir, df_F_measure_max, 'F1 of ITERATER-max')
    store_plot(plot_dir, df_g_mean_max, 'GM of ITERATER-max')
    store_plot(plot_dir, df_Balance_max, 'BPP of ITERATER-max')

    store_plot(plot_dir, df_F_measure_min, 'F1 of ITERATER-min')
    store_plot(plot_dir, df_g_mean_min, 'GM of ITERATER-min')
    store_plot(plot_dir, df_Balance_min, 'BPP of ITERATER-min')

    # store_plot(plot_dir, df_F_measure_max, 'F1 (Max. of V-Score for AL)')
    # store_plot(plot_dir, df_g_mean_max, 'GM (Max. of V-Score for AL)')
    # store_plot(plot_dir, df_Balance_max, 'BPP (Max. of V-Score for AL)')
    #
    # store_plot(plot_dir, df_F_measure_min, 'F1 (Min. of V-Score for AL)')
    # store_plot(plot_dir, df_g_mean_min, 'GM (Min. of V-Score for AL)')
    # store_plot(plot_dir, df_Balance_min, 'BPP (Min. of V-Score for AL)')

    # store_plot(plot_dir, df_F_measure_5_percent, 'F1_5%')
    # store_plot(plot_dir, df_F_measure_10_percent, 'F1_10%')
    # store_plot(plot_dir, df_F_measure_15_percent, 'F1_15%')
    # store_plot(plot_dir, df_F_measure_20_percent, 'F1_20%')
    # store_plot(plot_dir, df_g_mean_5_percent, 'GM_5%')
    # store_plot(plot_dir, df_g_mean_10_percent, 'GM_10%')
    # store_plot(plot_dir, df_g_mean_15_percent, 'GM_15%')
    # store_plot(plot_dir, df_g_mean_20_percent, 'GM_20%')
    # store_plot(plot_dir, df_Balance_5_percent, 'BPP_5%')
    # store_plot(plot_dir, df_Balance_10_percent, 'BPP_10%')
    # store_plot(plot_dir, df_Balance_15_percent, 'BPP_15%')
    # store_plot(plot_dir, df_Balance_20_percent, 'BPP_20%')


    # plt.rcParams['savefig.dpi'] = 900  # 图片像素
    # plt.rcParams['figure.figsize'] = (10.0, 10.0)
    # # plt.rcParams['figure.figsize'] = (20.0, 6.0)
    # df_F_measure_5_percent.plot.box(showmeans=True)
    # # df_universal.plot.box(title="Box Plot of GM values for OO Metrics with threshold")
    # plt.grid(linestyle="--", alpha=0.3)
    # # plt.ylim(0.2, 0.8)
    # # plt.xticks(rotation=0, fontsize=9.0)
    # plt.xticks(rotation=0, fontsize=18.0)
    # # plt.xticks(rotation=30, fontsize=15.0)
    # plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
    # # plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
    # y_major_locator = MultipleLocator(0.05)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.savefig(plot_dir + 'F_measure_5_percent.png')
    # plt.close()



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

    working_Directory = "F:/talcvdp/active_learning_middle_prior_pseudo_all_percents/"
    plot_Directory = "F:/talcvdp/active_learning_middle_prior_pseudo_all_percents/boxPlot_four_thresholds/"
    os.chdir(working_Directory)

    f1_gm_bpp_plot(working_Directory, plot_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")