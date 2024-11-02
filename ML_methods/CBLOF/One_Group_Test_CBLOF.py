import numpy as np
import pandas as pd
import argparse
import os
from ML_methods.utils.data import get_scaled, load_config, save_scaler, kalman_filter
from loguru import logger
from pyod.models.cblof import CBLOF
from ML_methods.utils.utils_def import  scaler_loss
from adtk.visualization import plot
import seaborn as sns



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
    df = pd.read_csv(TRAIN_FILE)
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
    group_df.to_csv(os.path.join(parent_directory, 'ML_methods','Reports_Methods' ,'Reports_CBLOF',
                                 DIR_EXP, 'csv_data', f'group_{i}.csv'),
                    index=False)
    scaled_group = get_scaled(group_df)
    group_list.append(scaled_group)
    save_scaler(scaled_group,
                os.path.join(parent_directory, 'ML_methods','Reports_Methods', 'Reports_CBLOF',
                             DIR_EXP, 'scaler_data', f'scaler_{i}.pkl'))
for i, group_data in enumerate(group_list):
    clf = CBLOF(use_weights=True)
    clf.fit(group_data)
    ###################
    cluster_labels = clf.labels_  # Cluster labels for each data point
    cluster_centers = clf.cluster_centers_
    y_train_scores = pd.DataFrame(index=group_data.index, columns=group_data.columns)

    # Convert cluster_centers to a DataFrame for easier indexing
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=group_data.columns)

    # Compute per-feature losses
    for idx in group_data.index:
        label = cluster_labels[idx]
        centroid = cluster_centers_df.loc[label]
        data_point = group_data.loc[idx]
        per_feature_loss = np.abs(data_point - centroid)
        y_train_scores.loc[idx] = per_feature_loss

        # Reset index and add timestamp
    y_train_scores.reset_index(inplace=True)
    y_train_scores.rename(columns={'index': 'timestamp'}, inplace=True)
    y_train_scores['timestamp'] = time_test.values
    y_train_scores.set_index('timestamp', inplace=True)
    #combined_loss = y_train_scores[group_data.columns].sum(axis=1)
    #df_combined_loss = pd.DataFrame({
        #'timestamp': y_train_scores['timestamp'],
        #'combined_loss': combined_loss
    #})
    #combined_loss.set_index('timestamp', inplace=True)


    ###################

    y_train_pred = clf.decision_scores_
    y_train_pred_proba = clf.predict_proba(group_data)

    df_timestamps = pd.DataFrame({'timestamp': time_})


    df_loss_final = pd.merge(y_train_scores, df_timestamps, on='timestamp', how='right').fillna(0)
    df_target_proba = pd.DataFrame(y_train_pred_proba, columns=['prob_1', 'prob_2'], index=time_test.index)
    df_target_proba['timestamp'] = time_test.values
    df_proba_final = pd.merge(df_target_proba, df_timestamps, on='timestamp', how='right').fillna(0)

    target_value, scalers_loss = scaler_loss(y_train_pred, 'cdf')
    df_target = pd.DataFrame({'target_value': target_value}, index=time_test.index)
    df_target['timestamp'] = time_test.values
    df_target_final = pd.merge(df_target, df_timestamps, on='timestamp', how='right').fillna(0)

    # Сохранение финальных данных
    output_dir = os.path.join(parent_directory, 'ML_methods','Reports_Methods', 'Reports_CBLOF', DIR_EXP)
    df_loss_final.to_csv(os.path.join(output_dir, 'csv_loss', f'loss_{i}.csv'), index=False)
    df_target_final.to_csv(os.path.join(output_dir, 'csv_predict', f'predict_{i}.csv'), index=False)
    df_proba_final.to_csv(os.path.join(output_dir, 'csv_predict_proba', f'predict_proba_{i}.csv'), index=False)
