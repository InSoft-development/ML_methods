from adtk.transformer import PcaReconstructionError
import pandas as pd
import argparse
import os
from ML_methods.utils.data import get_scaled, load_config, save_scaler, kalman_filter
from loguru import logger

from ML_methods.utils.utils_def import  scaler_loss

configs_train = ["dataset2", "dataset3","Sochi", "Yugres"]
current_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(current_directory, '..', '..'))
path_to_directory = parent_directory
config_path = os.path.join(path_to_directory, 'config', f'{configs_train[0]}.yml')
config = load_config(config_path)
MEAN_NAN = config['MEAN_NAN']
DROP_NAN = config['DROP_NAN']
ROLLING_MEAN = config['ROLLING_MEAN']
EXP_SMOOTH = config['EXP_SMOOTH']
DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']
KALMAN = config.get('KALMAN', False)
    #KKS = os.path.join(parent_directory, config['KKS'])
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
    parent_directory = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))
    TRAIN_FILE = os.path.join(parent_directory, config['TRAIN_FILE'])
    df = pd.read_csv(TRAIN_FILE, sep=',', parse_dates=['timestamp'])
    time_ = df['timestamp']
else:
    TRAIN_FILE = os.path.join(parent_directory, 'Reports', DIR_EXP,
                              'clear_data', 'clear_data.csv')
    df = pd.read_csv(TRAIN_FILE, sep=',', parse_dates=['timestamp'])
    df = df.drop(columns=['one_svm_value', 'check_index'])
KKS = os.path.join(parent_directory, config['KKS'])
# Предобработка данных
df = df[df[POWER_ID] > POWER_LIMIT]
if MEAN_NAN:
    df.fillna(df.mean(), inplace=True)
if DROP_NAN:
    df.dropna(inplace=True)
time_test = df['timestamp']
df.drop(columns='timestamp', inplace=True)
if KALMAN:
    df = kalman_filter(df)
if ROLLING_MEAN:
    df = df.rolling(window=ROLLING_MEAN_WINDOW).mean()

groups = pd.read_csv(KKS, sep=';')
logger.info(f"KKS: \n {groups}")
logger.info(f"Data: \n {df.head()}")

    # разбиение данных по группам, скейлинг и сохранение скейлеров
group_list = []
sum = 0
groups['group'] = groups['group'].astype(int)
logger.debug(f"KKS: \n {groups.dtypes}")
for i in range(NUM_GROUPS):
    group = groups[groups['group'] == i]
    logger.debug(group)
    if i != 0:
        group = pd.concat([group, groups[groups['group'] == 0]])
    if len(group) == 0:
        continue
    group_columns = group['kks'].tolist()
    group_df = df[group_columns]
    group_df.to_csv(os.path.join(parent_directory, 'ML_methods',
                                 'Reports_Methods' ,'Reports_PcaReconstructionError',
                                 DIR_EXP, 'csv_data', f'group_{i}.csv'),
                    index=False)
    scaled_group = get_scaled(group_df)
    group_list.append(scaled_group)
    save_scaler(scaled_group,
                os.path.join(parent_directory, 'ML_methods',
                             'Reports_Methods', 'Reports_PcaReconstructionError',
                             DIR_EXP, 'scaler_data', f'scaler_{i}.pkl'))
for i, group_data in enumerate(group_list):
    group_data.set_index(time_test, inplace=True)
    n_samples, n_features = group_data.shape
    max_k = min(n_samples, n_features)
    if max_k < 1:
        print(f"Недостаточно данных для выполнения PCA на группе {i}. Пропускаем эту группу.")
        continue
    k = min(2, max_k)
    clf = PcaReconstructionError(k=k)
    clf.fit(group_data)
    y_train_pred = clf.fit_transform(group_data)
    ###################
    y_train_scores = clf.transform(group_data, per_feature=True)

    df_timestamps = pd.DataFrame({'timestamp': time_})
    df_loss_final = pd.merge(y_train_scores, df_timestamps, on='timestamp', how='right').fillna(0)


    target_value, scalers_loss = scaler_loss(y_train_pred.astype(float), 'cdf')
    df_target = pd.DataFrame({'target_value': target_value}, index=time_test.index)
    df_target['timestamp'] = time_test.values
    df_target_final = pd.merge(df_target, df_timestamps, on='timestamp', how='right').fillna(0)

    # Сохранение финальных данных
    output_dir = os.path.join(parent_directory, 'ML_methods', 'Reports_Methods', 'Reports_PcaReconstructionError', DIR_EXP)
    df_loss_final.to_csv(os.path.join(output_dir, 'csv_loss', f'loss_{i}.csv'), index=False)

    df_target_final.to_csv(os.path.join(output_dir, 'csv_predict', f'predict_{i}.csv'), index=False)

