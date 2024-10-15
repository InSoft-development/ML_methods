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
configs_train = ["dataset2", "dataset3", "Sochi", "Yugres"]
# Путь к основной директории откуда запущен скрипт
current_dir = os.path.dirname(__file__)
#'C:\\Users\\dshteinberg\\PycharmProjects\\testsuite\\ML_methods'

# Путь к директории на уровень выше
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
#'C:\\Users\\dshteinberg\\PycharmProjects\\testsuite'

parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, default='dataset2')
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
    time_test = df['timestamp']
else:
    TRAIN_FILE = f'{parent_dir}/Reports/{DIR_EXP}/clear_data/clear_data.csv'
    df = pd.read_csv(TRAIN_FILE)
    df = df.drop(columns=['one_svm_value', 'check_index'])
df = df[df[POWER_ID] > POWER_LIMIT]
if MEAN_NAN:
    df = df.fillna(df.mean(), inplace=True)
if DROP_NAN:
    df = df.dropna()
    time = df['timestamp']
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
    print(group.shape)
    group.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_data/group_{i}.csv', index=False)
    group_list.append(get_scaled(group))
    save_scaler(group,f'{current_dir}/Reports_2/{DIR_EXP}/scaler_data/scaler_{i}.pkl')

clf_name = 'ECOD'
clf = ECOD()

for i in range(0, NUM_GROUPS):
    clf_name = 'ECOD'
    clf = ECOD()
    clf.fit(group_list[i])
    y_train_scores = clf.U_l
    y_train_scores_2 = clf.O
    y_train_pred = clf.predict_proba(group_list[i])
    # get outlier scores
    print(group_list[i].shape)
     #Проблема с размерностью
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_loss = pd.DataFrame(y_train_scores, columns=[group_list[i].columns])
    df_loss['timestamp'] = time
    df_loss = df_loss.dropna()
    print(df_loss.shape)
    df_loss = df_loss.drop(columns='timestamp')
    print(df_loss.shape)
    df_loss = pd.DataFrame(scaler.fit_transform(df_loss), columns=group_list[i].columns)
    print(df_loss.shape)
    df_loss['timestamp'] = time


    #target_value, scalers_loss = scaler_loss(df_loss['loss'], 'cdf')
    df_loss.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_loss/loss_{i}.csv', index=False)
    df_target = pd.DataFrame()
    df_target['target_value'] = y_train_pred
    df_target['timestamp'] = time

    df_target.to_csv(f'{current_dir}/Reports_2/{DIR_EXP}/csv_predict/predict_proba_{i}.csv', index=False)








if __name__ == '__main__':
    pass