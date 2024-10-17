import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
import joblib
from yaml import load
from yaml import FullLoader
from loguru import logger
from pykalman import KalmanFilter
from tqdm.auto import tqdm

def kalman_filter(df):
    # Создаем новый DataFrame для отфильтрованных данных
    filtered_df = pd.DataFrame(index=df.index)
    
    # Инициализируем прогресс-бар
    pbar = tqdm(total=len(df.columns), desc='Processing Columns')
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Инициализация фильтра Калмана
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Предполагаем, что данные в столбце - одномерные наблюдения
            measurements = df[column].values
            measurements = measurements.reshape(-1, 1) # Преобразуем в формат, подходящий для фильтра Калмана
            
            # Применяем фильтр к данным
            (filtered_state_means, _) = kf.filter(measurements)
            
            # Сохраняем отфильтрованные значения в новом DataFrame
            filtered_df[column] = filtered_state_means.flatten()
        else:
            # Если столбец не числовой, просто копируем его без изменений
            filtered_df[column] = df[column]
        
        # Обновляем прогресс-бар
        pbar.update(1)
    
    # Закрываем прогресс-бар после завершения цикла
    pbar.close()
    
    return filtered_df

def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)

def get_scaled(data_features):
  scaler = StandardScaler()
  scaled_data = pd.DataFrame(
      data=scaler.fit_transform(data_features), 
      columns=data_features.columns
  )
  return(scaled_data)

def save_scaler(data_features, save_path):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        data=scaler.fit_transform(data_features), 
        columns=data_features.columns
    )
    joblib.dump(scaler, save_path)


    
      

def get_len_size(LAG,x_size):
  return int(x_size/LAG) * LAG

def hist_threshold(df, percent):
    h,bins = np.histogram(pd.DataFrame(df['loss']),bins = 1000)
    max_p = sum(h)* percent
    s = 0
    N = -1
    for i in range(0,1000):
      s += h[i]
      if s > max_p:
        N = i 
        break
    return bins[N]

def get_interval(data,loss,df,time,count_anomaly,anomaly_treshold,left_space, right_space, left_history, right_history):
    anomaly_list = []
    time_list = []
    report_list = []
    data_list = []
    history_list = []
    i = 0
    logger.debug(f'porog {float(anomaly_treshold)}')
    for value in loss:
      if float(value)>anomaly_treshold:
        anomaly_list.append(value)
      else:
        if len(anomaly_list) > count_anomaly:

          report_list.append(df[i-len(anomaly_list):i])
          data_list.append(data[i-len(anomaly_list):i])

          if (i-len(anomaly_list)-left_history)>=0 and (i+right_history)<=len(df):
            history_list.append(data[i-len(anomaly_list)-left_history:i+right_history])
  
            logger.debug(f'all {len(data[i-len(anomaly_list)-left_space:i+right_space])}')
          elif (i-len(anomaly_list)-left_history)<0: 
            history_list.append(data[i-len(anomaly_list):i+right_history])
          
            logger.debug(f' left {len(data[i-len(anomaly_list):i+right_space])}')
            
          elif (i+right_history)>=len(df):
            history_list.append(data[i-len(anomaly_list)-left_history:i])
            
            logger.debug(f' right {len(data[i-len(anomaly_list)-left_space:i])}')
          else: 
            history_list.append(data[i-len(anomaly_list):i])
          time_list.clear()
          anomaly_list.clear()
        else:
          time_list.clear()
          anomaly_list.clear()
      
      i+=1
    return report_list, data_list, history_list
def get_anomaly_interval(loss, threshold,min_interval_len):
  interval_list = []
  loss_interval = []
  count = 0
  i = 0
  idx_list = []
  sum_anomaly = 0
  for val in loss:
    i+=1
    if val>threshold:
      loss_interval.append(val)
    else:
       count+=1
       loss_interval.append(val)
       if count>5:
         if len(loss_interval)>min_interval_len:
          interval_list.append(loss_interval)
          logger.info(f'Add anomaly interval, len {len(loss_interval)}')
          # idx_list.append((i-len(loss_interval),i))
          idx_list.append((loss.index[i-len(loss_interval)],loss.index[i]))
          # logger.debug((loss.index[i-len(loss_interval)],loss.index[i]))
          logger.debug(loss[i-len(loss_interval):i])
          sum_anomaly+=len(loss_interval)
         count = 0
         loss_interval.clear()
      
  logger.info(f'Sum anomaly {sum_anomaly}, part of anomaly {round(sum_anomaly/len(loss),3)}')
  # logger.info(f'')
  return interval_list, idx_list
