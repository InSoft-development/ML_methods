import yaml
import clickhouse_driver


client = clickhouse_driver.Client(user="default", password="asdf", host="10.23.23.32")
# client.execute('''
#     CREATE TABLE IF NOT EXISTS experiments (
#     id DateTime DEFAULT now(),
#     path String,
#     description String
# ) ENGINE = MergeTree() 
# ORDER BY id;
# ''')
from datetime import datetime

def insert_experiment(client, today_date, info):
    # Преобразуем today_date в timestamp (если это строка)
    if isinstance(today_date, str):
        timestamp = datetime.strptime(today_date, '%Y-%m-%d %H:%M:%S')
    elif isinstance(today_date, float):  # если это timestamp, конвертируем обратно в datetime
        timestamp = datetime.fromtimestamp(today_date)
    elif isinstance(today_date, datetime):
        timestamp = today_date
    else:
        raise ValueError("today_date должен быть строкой в формате 'YYYY-MM-DD HH:MM:SS', timestamp (float) или объектом datetime")

    # Подготавливаем данные для вставки
    path = today_date
    description = info

    # Выполняем вставку данных
    client.execute(
        "INSERT INTO experiments (id, path, description) VALUES", 
        [(timestamp, path, description)]
    )

    print("Данные успешно добавлены")

client = clickhouse_driver.Client(user="default", password="asdf", host="10.23.23.32")
today_date = datetime.now().strftime('%Y-%m-%d') +" "+ datetime.now().strftime('%H:%M:%S')
with open('Train_LSTM/config/settings/model.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
# info = []
# print(yaml_data['architecture'])
info = yaml_data['architecture']
# info['path'] = today_date
insert_experiment(client, today_date, info)


# Пример использования:
# client - экземпляр ClickHouse-клиента (например, от clickhouse_driver)
# today_date - строка в формате 'YYYY-MM-DD HH:MM:SS' или объект datetime
# info - словарь с полями 'path' и 'description'
