import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import argparse
import os
import shutil
from Train_LSTM.utils.data import get_scaled, load_config, save_scaler, kalman_filter
from loguru import logger

def dir_maker(DIR_EXP ):
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Создание директроии для проведения экспериметов
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', type=str, default='dataset2')
    parser.add_argument('--dir', type=str, default='C:\\Users\\dshteinberg\\PycharmProjects\\testsuite\\Train_LSTM')
    opt = parser.parse_args()
    try:
        os.mkdir(f'{current_dir}\\Reports_2\\{DIR_EXP}')
    except Exception as e:
        logger.error(e)
    try:
        os.mkdir(f'{current_dir}\\Reports_2\\{DIR_EXP}\\train_info\\')
    except Exception as e:
        logger.error(e)
    shutil.copy(f'{opt.dir}\\config\\{opt.station}.yml', f'{current_dir}\\Reports_2/{DIR_EXP}\\train_info/')
    try:
        os.mkdir(f'{current_dir}\\Reports_2\\{DIR_EXP}\\train_info\\model\\')
    except Exception as e:
        logger.error(e)
    try:
        os.mkdir(f'{current_dir}\\Reports_2\\{DIR_EXP}\\model_pt\\')
    except Exception as e:
        logger.error(e)
    try:
        os.mkdir(f'{current_dir}\\Reports_2\\{DIR_EXP}\\scaler_data\\')
    except Exception as e:
        logger.error(e)
    try:
        os.makedirs(f'{current_dir}\\Reports_2\\{DIR_EXP}\\csv_data\\')
    except Exception as e:
        logger.error(e)
    try:
        os.makedirs(f'{current_dir}\\Reports_2\\{DIR_EXP}\\csv_predict\\')
    except Exception as e:
        logger.error(e)
        try:
            os.makedirs(f'{current_dir}\\Reports_2\\{DIR_EXP}\\csv_loss\\')
        except Exception as e:
            logger.error(e)
    try:
        os.makedirs(f'{current_dir}\\Reports_2\\{DIR_EXP}\\scalers_loss\\')
    except Exception as e:
        logger.error(e)

def data_saver(data):
    pass


def scaler_loss(target_value, scaler_name, range_loss=100):
    if scaler_name == 'cdf':
        hist = np.histogram(target_value, bins=range_loss)
        # logger.debug(target_value)
        scaler_loss = scipy.stats.rv_histogram(hist)
        # logger.debug(hist)
        target_value = scaler_loss.cdf(target_value) * range_loss
        scaler_loss = hist
    elif scaler_name == 'minmax':
        scaler_loss = MinMaxScaler(feature_range=(0, range_loss))
        loss_2d = np.reshape(target_value, (-1, 1))
        scaler_loss.fit(loss_2d)
        target_value = scaler_loss.transform(loss_2d)
    return target_value, scaler_loss