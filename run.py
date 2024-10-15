import yaml
import pandas as pd
import os
import json
from loguru import logger
from Reports_variation.get_interval_streamlit import get_anomaly_interval_streamlit, rolling_probability,fill_zeros_with_last_value
from utils import count_files, extract_group_ids, find_indices_in_range, calculate_iou, union_intervals


# Определяем путь к файлу
file_path = 'config_station.yaml'

# Открываем файл и загружаем данные
with open(file_path, 'r') as file:
    config_data = yaml.safe_load(file)

# Определяем путь к файлу
file_path = 'config_exp.yaml'

# Открываем файл и загружаем данные
with open(file_path, 'r') as file:
    config_exp = yaml.safe_load(file)

DIR_EXP = "Reports/"
json_dir = "json_interval"
# os.makedirs(json_dir, exist_ok=True)


with open('analysis_report.txt', 'w') as report_file:
    for station_name, station_data in config_data['Station'].items():
        report_file.write(f"Station: {station_name}\n")
        find_anomaly_len = 0
        true_anomaly_len = 0
        logger.info(f"**********🔔 Processing {station_name}: **********")
        group_list = extract_group_ids(station_data)
        all_group_iou =[]
        for i in range(0,count_files(f"{station_data['predict']}")):
                
            loss_path = f"{station_data['loss']}/loss_{i}.csv"
            predict_path = f"{station_data['predict']}/predict_{i}.csv"
            data_path = f"{station_data['data']}"
            # print(loss_path)
            predict_df = pd.read_csv(predict_path)
            loss_df = pd.read_csv(loss_path)
            loss_df.index = pd.to_datetime(loss_df.index)
            # print("Индексы в формате datetime:", loss_df.index)
            data_df = pd.read_csv(data_path)
            rolled_df = rolling_probability(predict_df,config_exp['number_of_samples'],config_exp['roll_in_hours'])
            # get_anomaly_interval_streamlit(predict_df,)
            if len(rolled_df) != len(data_df):
                    time_df = pd.DataFrame()
                    time_df['timestamp'] = data_df['timestamp']
                    rolled_df = pd.merge(time_df, rolled_df, how='left', on='timestamp')
                    print(rolled_df)
            rolled_df.fillna(value={"target_value": 0}, inplace=True)
            fill_zeros_with_last_value(rolled_df)
            rolled_df.index = rolled_df['timestamp']
            rolled_df = rolled_df.drop(columns=['timestamp'])
            
            loss_df.index = loss_df['timestamp']
            loss_df = loss_df.drop(columns=['timestamp'])
            interval_list, idx_list = get_anomaly_interval_streamlit(rolled_df['target_value'],
                                                                    threshold_short=config_exp['threshold_short'],
                                                                    threshold_long=config_exp['threshold_long'],
                                                                    len_long=config_exp['len_long'],
                                                                    len_short=config_exp['len_short'],
                                                                    count_continue_short=config_exp['threshold_long'],
                                                                    count_continue_long=config_exp['threshold_long'],
                                                                    power=data_df[station_data['power_index']],
                                                                    power_limit=station_data['power_limit'])
            dict_list = []
            for j in idx_list:
                top_list = loss_df[j[0]:j[1]].mean().sort_values(ascending=False).index[:config_exp['count_top']].to_list()
                mean_measurement = list(loss_df[j[0]:j[1]].mean().sort_values(ascending=False).values[:config_exp['count_top']])
                report_dict = {
                    "time": (str(rolled_df.index[j[0]]), str(rolled_df.index[j[1]])),
                    "len": j[1] - j[0],
                    "index": j,
                    "top_sensors": top_list,
                    "measurement": mean_measurement
                }
                dict_list.append(report_dict)
            os.makedirs(f'{DIR_EXP}/{station_name.upper()}/{json_dir}/', exist_ok=True)
            
            with open(f'{DIR_EXP}/{station_name.upper()}/{json_dir}/{i}.json', "w") as outfile:
                    json.dump(dict_list, outfile, indent=4)
            # print(station_data["groups"])
            # print(group_list)
            if i in group_list:
                group_iou = []
                logger.info(f"    --> Processing Group {i}")
                target_intervals = next((group['intervals'] for group in station_data["groups"] if group['id'] == i), None)
                # print(target_intervals)
                time_ranges = [interval['time_range'] for interval in target_intervals]
                for t in time_ranges:
                    loss_df.index = pd.to_datetime(loss_df.index)
                    loss_df['Row_Number'] = range(len(loss_df))
                    # print(f'Time {t[0]} {t[1]}')
                    # start_end = loss_df.loc[t[0]:t[1]]
                    # print(f"Interval {start_end['Row_Number'][0]}, {start_end['Row_Number'][-1]}")
                    inter = find_indices_in_range(loss_df, t[0], t[1])
                    logger.info(f' True_interval {inter}')
                    # print()
                    # Загружаем JSON-файл
                    file_path = f'{DIR_EXP}/{station_name.upper()}/{json_dir}/{i}.json'
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    # Функция для нахождения пересекающегося документа
                    def find_document(data, start, end):
                        result = []
                        for doc in data:
                            doc_start, doc_end = doc['index']
                            if doc_start <= end and doc_end >= start:
                                result.append(doc)
                        return result
                    # print(data)
                    documents = find_document(data, inter[0], inter[1])
                    # Выводим результаты
                    intervals = union_intervals(documents)
                    iou = calculate_iou(inter, intervals)
                    true_anomaly_len = true_anomaly_len + (inter[1]-inter[0])
                    find_anomaly_len = find_anomaly_len + (intervals[1]-intervals[0])
                    logger.debug(true_anomaly_len)
                    logger.debug(find_anomaly_len)
                    group_iou.append(iou)
                    logger.info(f"The Intersection over Union (IoU) is: {iou}")       
                mean_iou_group = sum(group_iou)/len(group_iou)
                logger.info(f"Mean IoU group {i}: {mean_iou_group}")
                report_file.write(f"Mean IoU group {i}: {mean_iou_group}\n")
                all_group_iou.append(mean_iou_group)
        if all_group_iou:  
            mean_iou_all = sum(all_group_iou)/len(all_group_iou)
            logger.info(f"Station {station_name} IoU: {mean_iou_all}")
        part_of_find_anomaly = (find_anomaly_len/true_anomaly_len) * 100
        logger.info(f"All persent find anomaly: {part_of_find_anomaly}")  
        
        report_file.write(f"Mean IoU: {mean_iou_all:.2f}\n")
        report_file.write(f"Percentage of detected anomalies: {part_of_find_anomaly:.2f}%\n")
        report_file.write("----------------------------\n")
        
