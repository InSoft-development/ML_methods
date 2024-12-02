import os
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from ML_methods.utils.data import get_scaled, load_config, save_scaler, kalman_filter
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

configs_train = ["dataset2", "dataset3","Sochi", "Yugres"]
current_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(current_directory, '..', '..'))
def process_station(station_name, path_to_directory, current_directory, parent_directory):
    config_path = os.path.join(path_to_directory, 'config', f'{station_name}.yml')
    config = load_config(config_path)
    MEAN_NAN = config['MEAN_NAN']
    DROP_NAN = config['DROP_NAN']
    ROLLING_MEAN = config['ROLLING_MEAN']
    KALMAN = config.get('KALMAN', False)
    #KKS = os.path.join(parent_directory, config['KKS'])
    NUM_GROUPS = config['NUM_GROUPS']
    DIR_EXP = config['DIR_EXP']
    POWER_ID = config['POWER_ID']
    POWER_LIMIT = config['POWER_LIMIT']
    ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
    USE_ALL_DATA = config['USE_ALL_DATA']

    # Загрузка данных
    if USE_ALL_DATA:
        parent_directory = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))
        TRAIN_FILE = os.path.join(parent_directory, config['TRAIN_FILE'])
        df = pd.read_csv(TRAIN_FILE, sep=',', parse_dates=['timestamp'])
        time_ = df['timestamp']
    else:
        TRAIN_FILE = os.path.join(parent_directory, 'Reports',
                                  DIR_EXP, 'clear_data', 'clear_data.csv')
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
    groups['group'] = groups['group'].astype(int)
    # Обработка групп
    group_list = []
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
                                     'Window_reports', 'Reports_IncPCA',
                                     DIR_EXP, 'csv_data', f'group_{i}.csv'), index=False)
        scaled_group = get_scaled(group_df)
        group_list.append(scaled_group)
        save_scaler(scaled_group,
                    os.path.join(parent_directory, 'ML_methods',
                                 'Window_reports', 'Reports_IncPCA',
                                 DIR_EXP, 'scaler_data', f'scaler_{i}.pkl'))

    window_size = 10
    step_size = 1
    for i, group_data in enumerate(group_list):
        # Сброс индекса для обеспечения числовой индексации
        group_data.reset_index(drop=True, inplace=True)
        n_samples, n_features = group_data.shape
        df_timestamps = pd.DataFrame({'timestamp': time_})
        # Создание перекрывающихся окон
        windows = create_windows(group_data, window_size, step_size)
        time_windows = create_windows(time_test.reset_index(drop=True), window_size, step_size)

        # Инициализация модели Incremental PCA
        n_components = min(n_features, window_size - 1)  # Количество компонент
        ipca = IncrementalPCA(n_components=n_components)

        # Инициализация аккумуляторов для ошибок восстановления и счетчиков
        accumulated_errors_per_feature = np.zeros((n_samples, n_features))
        counts = np.zeros(n_samples)

        # Инкрементальное обучение модели и вычисление ошибок восстановления
        for window_start, (window_data, time_window) in enumerate(zip(windows, time_windows)):
            # Инкрементальное обучение модели
            ipca.partial_fit(window_data)

            # Восстановление данных
            transformed_data = ipca.transform(window_data)
            reconstructed_data = ipca.inverse_transform(transformed_data)

            # Вычисление ошибки восстановления по каждой фиче для каждого наблюдения в окне
            error_per_feature = (window_data - reconstructed_data) ** 2  # Размерность: (window_size, n_features)

            # Накопление ошибок и счетчиков для каждого наблюдения
            indices = np.arange(window_start, window_start + window_size)
            accumulated_errors_per_feature[indices] += error_per_feature
            counts[indices] += 1

        # Избежание деления на ноль
        counts[counts == 0] = 1

        # Вычисление средней ошибки восстановления по каждой фиче для каждого наблюдения
        average_errors_per_feature = accumulated_errors_per_feature / counts[:,
                                                                      np.newaxis]  # Размерность: (n_samples, n_features)

        # Создание df_loss с ошибками по фичам и временными метками
        df_loss = pd.DataFrame(average_errors_per_feature, columns=group_data.columns, index=time_test)
        df_loss_final = pd.merge(df_loss, df_timestamps, on='timestamp', how='right').fillna(0)

        # Вычисление общей ошибки восстановления для каждого наблюдения
        reconstruction_error = average_errors_per_feature.mean(axis=1)  # Средняя ошибка по фичам для каждого наблюдения

        # Вычисление степени аномальности (anomaly score)
        anomaly_score = (reconstruction_error - reconstruction_error.mean()) / reconstruction_error.std()
        anomaly_score = np.clip(anomaly_score, a_min=0, a_max=None)  # Обеспечение неотрицательных значений
        target_value, scalers_loss = scaler_loss(anomaly_score, 'cdf')
        # Создание df_target с степенью аномальности и временными метками
        df_target = pd.DataFrame({
            'target_value': target_value
        }, index=time_test)

        df_target_final = pd.merge(df_target, df_timestamps, on='timestamp', how='right').fillna(0)
        output_dir = os.path.join(parent_directory, 'ML_methods', 'Window_reports', 'Reports_IncPCA', DIR_EXP)
        df_loss_final.to_csv(os.path.join(output_dir, 'csv_loss', f'loss_{i}.csv'), index=False)

        df_target_final.to_csv(os.path.join(output_dir, 'csv_predict', f'predict_{i}.csv'), index=False)

path_to_directory = parent_directory
for station in configs_train:
    logger.info(f"Обработка станции {station}")
    try:
        process_station(station, path_to_directory, current_directory, parent_directory)
    except Exception as e:
        logger.error(f"Ошибка при обработке станции {station}: {e}")