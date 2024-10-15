import subprocess
import os
import shutil
import yaml
import clickhouse_driver

import os
import shutil
from datetime import datetime

client = clickhouse_driver.Client(user="default", password="asdf", host="10.23.23.32")
today_date = datetime.now().strftime('%Y-%m-%d') +" "+ datetime.now().strftime('%H:%M:%S')

def insert_experiment(client, today_date, info, archive_directory):
    if isinstance(today_date, str):
            timestamp = datetime.strptime(today_date, '%Y-%m-%d %H:%M:%S')
    elif isinstance(today_date, float):  # если это timestamp, конвертируем обратно в datetime
        timestamp = datetime.fromtimestamp(today_date)
    elif isinstance(today_date, datetime):
        timestamp = today_date
    else:
        raise ValueError("today_date должен быть строкой в формате 'YYYY-MM-DD HH:MM:SS', timestamp (float) или объектом datetime")
    current_directory = os.path.abspath(os.getcwd())
    # Подготавливаем данные для вставки
    path = f"{current_directory}/{archive_directory}/{today_date}"
    description = info

    # Выполняем вставку данных
    client.execute(
        "INSERT INTO experiments (id, path, description) VALUES", 
        [(timestamp, path, description)]
    )

    print("Данные успешно добавлены")

def copy_folders_with_date(source_dir, destination_dir, today_date):
    """
    Копирует все папки из source_dir в destination_dir,
    создавая папку с текущей датой в destination_dir.

    Аргументы:
    source_dir (str): Путь к исходной директории, из которой копируются папки.
    destination_dir (str): Путь к целевой директории, в которую копируются папки.

    # Возвращает:
    # None
    # """
    # Создаем директорию с текущей датой в destination_dir
   
    # current_time = datetime.now().strftime('%H:%M:%S')
    # today_date = today_date + currnt
    destination_with_date = os.path.join(destination_dir, today_date)
    if not os.path.exists(destination_with_date):
        os.makedirs(destination_with_date)

    # Копируем каждую папку из source_dir в destination_with_date
    for folder_name in os.listdir(source_dir):
        source = os.path.join(source_dir, folder_name)
        destination = os.path.join(destination_with_date, folder_name)
        if os.path.isdir(source):
            shutil.copytree(source, destination)
            print(f'Папка {folder_name} скопирована в директорию {destination_with_date}')
        else:
            print(f'{folder_name} не является папкой и не будет скопирован')

    print(f'Все папки из {source_dir} скопированы в {destination_with_date}')
    

# Пример использования функции:
source_directory = 'Reports'
archive_directory = 'Archive'
with open('Train_LSTM/config/settings/model.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
with open('analysis_report.txt', 'r') as file:
    data = file.read()
info = yaml_data['architecture'] +  '\n' + data


copy_folders_with_date(source_directory, archive_directory,today_date)
insert_experiment(client, today_date, info, archive_directory)

# config_process = ["T"]
configs_train = ["dataset3", "dataset2", "Sochi", "Yugres"]
configs_offline = ["offline-LSTM/config/data2.yaml",
                   "offline-LSTM/config/data3.yaml",
                   "offline-LSTM/config/sochi.yaml",                  
                   "offline-LSTM/config/yugres.yaml"]

CLEAN = False
TRAIN = True
window = True

if CLEAN:
# Запуск train.py с разными конфигурациями
    for config in configs_train:
        subprocess.run(["python", "Train_LSTM/clear_data.py", "--station", config, "--dir", "Train_LSTM"])
if TRAIN:
    if window:
        for config in configs_train:
            subprocess.run(["python", "Train_LSTM/train_window.py", "--station", config, "--dir", "Train_LSTM"])
    else:
# Запуск train.py с разными конфигурациями
        for config in configs_train:
            subprocess.run(["python", "Train_LSTM/train.py", "--station", config, "--dir", "Train_LSTM"])



    # Запуск predict_ofline.py с разными конфигурациями
if window:
    for config in configs_offline:
        subprocess.run(["python", "offline-LSTM/predict_offline.py", "--config_path", config, "--csv_kks", "True", "--file_format", "csv" ])
else:
    # Запуск predict_ofline.py с разными конфигурациями
    for config in configs_offline:
        subprocess.run(["python", "offline-LSTM/predict_offline_window.py", "--config_path", config, "--csv_kks", "True", "--file_format", "csv" ])
# Запуск run.py
subprocess.run(["python", "run.py"])