from sklearn.preprocessing import MinMaxScaler

from data import *
import pandas as pd
import argparse
import os
import shutil
from ML_methods.utils.data import get_scaled, load_config, save_scaler, kalman_filter
from loguru import logger
from pyod.models.ecod import ECOD
from utils_def import  scaler_loss
configs_train = ["dataset2", "Sochi", "Yugres"]
# Путь к основной директории откуда запущен скрипт
current_dir = os.path.dirname(__file__)
#'C:\\Users\\dshteinberg\\PycharmProjects\\testsuite\\ML_methods'

# Путь к директории на уровень выше
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
#'C:\\Users\\dshteinberg\\PycharmProjects\\testsuite'
for i in configs_train:
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', type=str, default=f'{i}')
    parser.add_argument('--dir', type=str, default='C:\\Users\\dshteinberg\\PycharmProjects\\testsuite\\Train_LSTM')
    opt = parser.parse_args()
    config = load_config(f'{opt.dir}/config/{opt.station}.yml')
    MEAN_NAN = config['MEAN_NAN']
    DROP_NAN = config['DROP_NAN']

    ROLLING_MEAN = config['ROLLING_MEAN']
    EXP_SMOOTH = config['EXP_SMOOTH']
    DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']

    KKS = f"{parent_dir}/{config['KKS']}"
    NUM_GROUPS = config['NUM_GROUPS']
    LAG = config['LAG']
    DIR_EXP = config['DIR_EXP']

    EPOCHS = config['EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    POWER_ID = config['POWER_ID']
    POWER_LIMIT = config['POWER_LIMIT']
    ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
    USE_ALL_DATA = config['USE_ALL_DATA']
    if USE_ALL_DATA:
        TRAIN_FILE = f"{parent_dir}/{config['TRAIN_FILE']}"
        df = pd.read_csv(TRAIN_FILE, sep=',')
        time_ = df['timestamp']
    else:
        TRAIN_FILE = f'{parent_dir}/Reports/{DIR_EXP}/clear_data/clear_data.csv'
        df = pd.read_csv(TRAIN_FILE)
        df = df.drop(columns=['one_svm_value', 'check_index'])
    df = df[df[POWER_ID] > POWER_LIMIT]
    if MEAN_NAN:
        df = df.fillna(df.mean(), inplace=True)
    if DROP_NAN:
        df = df.dropna()
        time_test = df['timestamp']
        df = df.drop(columns='timestamp')
    KALMAN = False
    if KALMAN:
        df = kalman_filter(df)
    if ROLLING_MEAN:
        rolling_mean = df.rolling(window=ROLLING_MEAN_WINDOW).mean()

    groups = pd.read_csv(KKS, sep=';')
    logger.info(f"KKS: \n {groups}")
    logger.info(f"Data: \n {df.head()}")


    # разбиение данных по группам, скейлинг и сохранение скейлеров
    group_list = []
    sum = 0
    groups['group'] = groups['group'].astype(int)
    logger.debug(f"KKS: \n {groups.dtypes}")
    for i in range(0, NUM_GROUPS):
        group = groups[groups['group'] == i]
        logger.debug(group)
        if i != 0:
            group = group._append(groups[groups['group'] == 0])
        sum += len(group)
        if len(group) == 0:
            continue
        group = df[group['kks']]
        group.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_data/group_{i}.csv', index=False)
        group_list.append(get_scaled(group))
        save_scaler(group,f'{current_dir}/Reports_2/{DIR_EXP}/scaler_data/scaler_{i}.pkl')

    clf_name = 'ECOD'
    clf = ECOD()

    for i in range(0, NUM_GROUPS):
        clf_name = 'ECOD'
        clf = ECOD()
        clf.fit(group_list[i])

        y_train_scores = clf.U_l  # csv_loss
        y_train_scores_2 = clf.O  # csv_loss_ver_O_
        y_train_pred = clf.decision_scores_  # csv_predict
        y_train_pred_proba = clf.predict_proba(group_list[i])  # csv_predict_proba



        df_timestamps = pd.DataFrame()
        df_timestamps['timestamp'] = time_

        # scaler = MinMaxScaler(feature_range=(0, 100))
        df_loss = pd.DataFrame(y_train_scores, columns=group_list[i].columns, index=time_test.index)
        df_loss_2 = pd.DataFrame(y_train_scores_2, columns=group_list[i].columns, index=time_test.index)  #
        # df_loss = pd.DataFrame(scaler.fit_transform(df_loss), columns=group_list[i].columns)
        # df_loss_2 = pd.DataFrame(scaler.fit_transform(df_loss_2), columns=group_list[i].columns)
        df_loss['timestamp'] = time_test
        df_loss_2['timestamp'] = time_test
        df_loss_final = pd.merge(df_loss, df_timestamps, on='timestamp', how='right')
        df_loss_final.fillna(0, inplace=True)
        df_loss_final_2 = pd.merge(df_loss_2, df_timestamps, on='timestamp', how='right')
        df_loss_final_2.fillna(0, inplace=True)
        # target_value, scalers_loss = scaler_loss(df_loss['loss'], 'cdf')

        df_target_proba = pd.DataFrame(y_train_pred_proba, columns=['prob_1', 'prob_2'], index=time_test.index)
        df_target_proba['timestamp'] = time_test
        df_proba_final = pd.merge(df_target_proba, df_timestamps, on='timestamp', how='right')
        df_proba_final.fillna(0, inplace=True)

        df_target = pd.DataFrame()
        df_target = pd.DataFrame(y_train_pred, columns=['target_value'])
        target_value, scalers_loss = scaler_loss(df_target['target_value'], 'cdf')
        df_target = pd.DataFrame()
        df_target = pd.DataFrame(target_value, columns=['target_value'], index=time_test.index)
        df_target['timestamp'] = time_test
        df_target_final = pd.merge(df_target, df_timestamps, on='timestamp', how='right')
        df_target_final.fillna(0, inplace=True)

        # save final data
        df_loss_final.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_loss/loss_{i}.csv', index=False)
        df_loss_final_2.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_loss_ver_O_/loss_ver_O_{i}.csv', index=False)
        df_target_final.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_predict/predict_{i}.csv', index=False)
        df_proba_final.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_predict_proba/predict_proba_{i}.csv', index=False)
