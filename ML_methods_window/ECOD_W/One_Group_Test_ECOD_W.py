import os

import numpy as np
import pandas as pd
from ML_methods.utils.data import get_scaled, load_config, save_scaler, kalman_filter
from loguru import logger
from pyod.models.ecod import ECOD
from ML_methods.utils.utils_def import scaler_loss



def create_windows(data, window_size, step_size):
    """
    Функция для разделения данных на окна.
    :param data: Массив данных для разделения.
    :param window_size: Размер каждого окна (количество временных шагов).
    :param step_size: Шаг смещения между окнами.
    :return: Массив данных, разделенных на окна.
    """
    sequences = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        seq = data[start:end]
        sequences.append(seq)
    return np.array(sequences)



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
    scaled_group = get_scaled(group_df)
    group_list.append(scaled_group)

for i, group_data in enumerate(group_list):
    X_train = group_data.to_numpy()
    N, num_features = X_train.shape
    window_size = 10
    step_size = 1
    X_train_windows = create_windows(X_train, window_size=window_size, step_size=step_size)

    # Initialize accumulators
    accumulated_U_l = np.zeros((N, num_features))
    accumulated_O = np.zeros((N, num_features))
    accumulated_decision_scores = np.zeros(N)
    accumulated_proba = np.zeros((N, 2))
    counts = np.zeros(N)

    # Process each window
    for window_idx, start in enumerate(range(0, N - window_size + 1, step_size)):
        end = start + window_size
        window_data = X_train[start:end]
        clf = ECOD()
        clf.fit(window_data)
        y_train_scores = clf.U_l  # (window_size, num_features)
        y_train_scores_2 = clf.O  # (window_size, num_features)
        y_train_pred = clf.decision_scores_  # (window_size,)
        y_train_pred_proba = clf.predict_proba(window_data)  # (window_size, 2)

        # Accumulate outputs
        accumulated_U_l[start:end] += y_train_scores
        accumulated_O[start:end] += y_train_scores_2
        accumulated_decision_scores[start:end] += y_train_pred
        accumulated_proba[start:end] += y_train_pred_proba
        counts[start:end] += 1

    # Avoid division by zero
    counts[counts == 0] = 1

    # Average the accumulated outputs
    final_U_l = accumulated_U_l / counts[:, None]
    final_O = accumulated_O / counts[:, None]
    final_decision_scores = accumulated_decision_scores / counts
    final_proba = accumulated_proba / counts[:, None]

    # Proceed to create dataframes as in your original code
    y_train_scores = final_U_l
    y_train_scores_2 = final_O
    y_train_pred = final_decision_scores
    y_train_pred_proba = final_proba

    # Ensure time_test has the correct length
    time_test = time_test.iloc[:N]
    df_timestamps = pd.DataFrame({'timestamp': time_})

    df_loss = pd.DataFrame(y_train_scores, columns=group_data.columns, index=time_test.index)
    df_loss_2 = pd.DataFrame(y_train_scores_2, columns=group_data.columns, index=time_test.index)
    df_loss['timestamp'] = time_test.values
    df_loss_2['timestamp'] = time_test.values
    df_loss_final = pd.merge(df_loss, df_timestamps, on='timestamp', how='right').fillna(0)
    df_loss_final_2 = pd.merge(df_loss_2, df_timestamps, on='timestamp', how='right').fillna(0)

    df_target_proba = pd.DataFrame(y_train_pred_proba, columns=['prob_1', 'prob_2'], index=time_test.index)
    df_target_proba['timestamp'] = time_test.values
    df_proba_final = pd.merge(df_target_proba, df_timestamps, on='timestamp', how='right').fillna(0)

    target_value, scalers_loss = scaler_loss(y_train_pred, 'cdf')
    df_target = pd.DataFrame({'target_value': target_value}, index=time_test.index)
    df_target['timestamp'] = time_test.values
    df_target_final = pd.merge(df_target, df_timestamps, on='timestamp', how='right').fillna(0)

    # Save the final dataframes
    output_dir = os.path.join(parent_directory, 'ML_methods', 'Window_reports', 'Reports_ECOD', DIR_EXP)


    df_loss_final.to_csv(os.path.join(output_dir, 'csv_loss', f'loss_{i}.csv'), index=False)
    df_loss_final_2.to_csv(os.path.join(output_dir, 'csv_loss_ver_O_', f'loss_ver_O_{i}.csv'), index=False)
    df_target_final.to_csv(os.path.join(output_dir, 'csv_predict', f'predict_{i}.csv'), index=False)
    df_proba_final.to_csv(os.path.join(output_dir, 'csv_predict_proba', f'predict_proba_{i}.csv'), index=False)

    print(f'Processed group {i}')