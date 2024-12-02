import os
import pandas as pd
import numpy as np
from river import anomaly
from ML_methods.utils.data import get_scaled, load_config, save_scaler, kalman_filter, get_scaled_2
from loguru import logger
from ML_methods.utils.utils_def import scaler_loss

def create_windows(data, window_size, step_size):
    """
    Function to split data into overlapping windows.
    :param data: DataFrame to split.
    :param window_size: Size of each window (number of time steps).
    :param step_size: Shift between windows.
    :return: List of DataFrames, each representing a window.
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end]
        windows.append(window)
    return windows

configs_train = ["dataset2", "dataset3", "Sochi", "Yugres"]
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

# Data preprocessing
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

# Splitting data into groups, scaling, and saving scalers
group_list = []
groups['group'] = groups['group'].astype(int)
logger.debug(f"KKS: \n {groups.dtypes}")

for i in range(NUM_GROUPS):
    group = groups[groups['group'] == i]
    logger.debug(group)
    if i != 0:
        group = pd.concat([group, groups[groups['group'] == 0]])
    if len(group) == 0:
        print(f"No data for group {i}. Skipping this group.")
        continue
    group_columns = group['kks'].tolist()
    group_df = df[group_columns]
    group_df.to_csv(os.path.join(parent_directory, 'ML_methods',
                                 'Window_reports', 'Reports_HalfSpaceTrees',
                                 DIR_EXP, 'csv_data', f'group_{i}.csv'), index=False)

    scaled_group = get_scaled_2(group_df)
    group_list.append(scaled_group)
    save_scaler(scaled_group,
                os.path.join(parent_directory, 'ML_methods',
                             'Window_reports', 'Reports_HalfSpaceTrees',
                             DIR_EXP, 'scaler_data', f'scaler_{i}.pkl'))


window_size = 10
step_size = 1

for i, group_data in enumerate(group_list):
    df_timestamps = pd.DataFrame({'timestamp': time_})
    # Reset index to ensure numerical indexing
    group_data.reset_index(drop=True, inplace=True)
    n_samples, n_features = group_data.shape
    if n_samples == 0:
        print(f"No data for group {i}. Skipping this group.")
        continue

    # Initialize the Half-Space Trees model
    model = anomaly.HalfSpaceTrees(
        n_trees=25,
        height=15,
        window_size=100,
        seed=42
    )

    accumulated_scores = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    accumulated_feature_losses = np.zeros((n_samples, n_features))

    # Iterate over each observation
    for index, row in group_data.iterrows():
        print(row)
        x = row.to_dict()

        # Get the anomaly score
        score = model.score_one(x)
        # Get per-feature loss
        feature_losses = model.score_features_one(x)
        # Update the model
        model.learn_one(x)

        # Accumulate the scores
        accumulated_scores[index] += score
        counts[index] += 1

        # Accumulate the feature losses
        feature_loss_values = [feature_losses.get(feature, 0.0) for feature in group_data.columns]
        accumulated_feature_losses[index] += feature_loss_values

    # Avoid division by zero
    counts[counts == 0] = 1

    # Compute average anomaly scores
    average_scores = accumulated_scores / counts
    target_value, scalers_loss = scaler_loss(average_scores, 'cdf')
    # Create df_target with anomaly scores
    df_target = pd.DataFrame({
        'target_value': target_value
    }, index=time_test)
    df_target_final = pd.merge(df_target, df_timestamps, on='timestamp', how='right').fillna(0)
    # Normalize anomaly scores if needed

    # Compute average feature losses
    average_feature_losses = accumulated_feature_losses / counts[:, np.newaxis]

    # Create df_loss with per-feature losses
    df_loss = pd.DataFrame(average_feature_losses, columns=group_data.columns, index=time_test)
    df_loss_final = pd.merge(df_loss, df_timestamps, on='timestamp', how='right').fillna(0)
    output_dir = os.path.join(parent_directory, 'ML_methods', 'Window_reports', 'Reports_HalfSpaceTrees', DIR_EXP)
    df_loss_final.to_csv(os.path.join(output_dir, 'csv_loss', f'loss_{i}.csv'), index=False)

    df_target_final.to_csv(os.path.join(output_dir, 'csv_predict', f'predict_{i}.csv'), index=False)


