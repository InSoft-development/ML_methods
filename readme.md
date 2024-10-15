## Запуск проекта
1. Подготовка файлов и конфига **config_station.yaml**: 
* Файл с данными slices.csv
* Файл kks для разбиения на группы kks_with_groups.csv
* Указать путь для сохранения предсказаний таргетной функции Reports/**название набора**/csv_predict
* Указать путь для сохранения ошибок для каждого датчика   Reports/**название набора**/csv_loss
* Указать значения датчика мощности 
* Указать лимит значения мощности
* Указать интервалы и датчики в которых была известа аномалия 

Пример заполнения:
```yaml
Station:
  dataset2:
    data: data/dataset2/slices.csv
    kks: data/dataset2/kks_with_groups.csv
    predict: Reports/DATASET2/csv_predict
    loss: Reports/DATASET2/csv_loss
    power_index: MKA01CE903
    power_limit: 100
    groups:
      - id: 2
        intervals:
          - kks_list: [MAD20CT148, MAD20CT148A, MKU01CT001]
            time_range: ["2022-03-29 03:40:00", "2022-04-03 03:40:00"]
          - kks_list: [MKU02CT001]
            time_range: ["2022-05-08 03:40:00", "2022-05-10 03:40:00"]
          - kks_list: [MKU02CT001]
            time_range: ["2022-05-17 16:00:00", "2022-05-22 16:00:00"]
      - id: 3
        intervals:
          - kks_list: [MAV30CT001]
            time_range: ["2022-03-29 03:40:00", "2022-04-03 03:40:00"]
```



1. Настройка тренировки моделей: 
В файле конфига выбрать модель ***Train_LSTM/settings/model.yaml***
```yaml
architecture: "simple_autoencoder"
# Можно выбрать из: simple_autoencoder bidirectional_lstm conv_autoencoder,transformer_autoencoder
 # в папке model.py можно менять параметры
 ```
 Создать конфиг для обучения в ***Train_LSTM/settings/model.yaml***  с названием станции например ***dataset2.yaml***
 ```yaml
 # Параметры преобразования данных
MEAN_NAN: False # Строки со значением Nan заменяются стредними значениями столбца
DROP_NAN: True # Строки со значением Nan удаляются
ROLLING_MEAN: False
EXP_SMOOTH: False
DOUBLE_EXP_SMOOTH: False
ROLLING_MEAN_WINDOW: 32
AUTO_DROP_LIST: False
DROP_LIST: [] #удаление датчиков из отчета CSV

# Параметры данных
USE_ALL_DATA: True
POWER_ID: 'MKA01CE903'
POWER_LIMIT: 100
TRAIN_FILE: 'data/dataset2/slices.csv'
KKS: 'data/dataset2/kks_with_groups_new.csv' #Файл с группами
NUM_GROUPS: 4 #Количество групп

# Параметры обучения модели
DIR_EXP: 'DATASET2' # Название директории в котрый сохраняются отчеты и веса(все хранится в корневой папке Reports)
EPOCHS: 1
BATCH_SIZE: 1024
LAG: 1 #Значение лага обучения

# Параметры очистки 
INTERVAL_REQ: [[20,30],[30,40]]
INTERVAL_CLEAR_LEN: 5000
INTERVAL_ANOMALY_LEN: 10
DISCOUNT_BETWEEN_IDX: 15
```
2. Настройка анализа тестовых данных:
Создать и заполнить файл конфигурации в ***offline-LSTM/config***, например ***data2.yaml***:
```yaml
#Преобразование входных данных
ROLLING_MEAN: False
EXP_SMOOTH: False
DOUBLE_EXP_SMOOTH: False
ROLLING_MEAN_WINDOW: 32

# Входные параметры
KKS: 'data/dataset2/kks_with_groups_new.csv' #Файл с группами
WEIGHTS: 'Reports/DATASET2/model_pt'
SCALER: 'Reports/DATASET2/scaler_data'

CSV_SAVE_RESULT_PREDICT: 'Reports/DATASET2/csv_predict/'
CSV_SAVE_RESULT_LOSS: 'Reports/DATASET2/csv_loss/'
CSV_DATA: 'Reports/DATASET2/csv_data/'
JSON_DATA: 'Reports/DATASET2/json_interval/'
SCALER_LOSS: 'Reports/DATASET2/scaler_loss/'
# TEST_FILE: SELECT * FROM slices order by timestamp limit 1
TEST_FILE: 'data/dataset2/slices.csv'

# Параметры алгоритма
POWER_ID: 'MKA01CE903'
POWER_LIMIT: 100
POWER_FILL: True
COUNT_NEXT: 288 #количество отсчетов отслеживания мощности при заливке
NUM_GROUPS: 4 #Количество групп
LAG: 1 #Значение лага обучения
MEAN_NAN: False
DROP_NAN: True
AUTO_DROP_LIST: False # удаление датчиков нулевой группы
DROP_LIST: [] #удаление датчиков из отчета CSV


ROLLING_MEAN_LOSS: 128
SCALER_LOSS_NAME: 'cdf'  # или minmax
# ANOMALY_TRESHOLD: 0.8
LEN_LONG_ANOMALY: 800
LEN_SHORT_ANOMALY: 300
COUNT_CONTINUE_SHORT: 10
COUNT_CONTINUE_LONG: 15
SHORT_TRESHOLD: 98
LONG_TRESHOLD: 97

COUNT_TOP: 3
```
3. Добавить названия файлов для обучения и анализа  в файле ***start_pipeline.py***
```python
# config_process = ["T"]
configs_train = ["dataset3", "dataset2", "Sochi", "Yugres"]
configs_offline = ["offline-LSTM/config/data2.yaml",
                   "offline-LSTM/config/data3.yaml",
                   "offline-LSTM/config/sochi.yaml",                  
                   "offline-LSTM/config/yugres.yaml"]
```
Запустить файл:
```bash
python run.py
```

## Описание Алгоритма

### Загрузка Конфигурации

- **Конфигурация Станций (`config_station.yaml`)**: Скрипт загружает настройки, которые определяют пути к данным предсказаний и источникам данных для каждой станции.
- **Конфигурация Эксперимента (`config_exp.yaml`)**: Загружаются параметры сглаживания, пороги и другие экспериментальные настройки.

### Подготовка Директории

- Создается папка `json_interval`, если она ещё не существует, для сохранения результатов анализа в формате JSON.

### Обработка Данных по Станциям

- Для каждой станции из конфигурации извлекается список групп идентификаторов.
- Осуществляется перебор файлов с данными о потерях (`loss`) и предсказаниях (`predict`).

### Анализ Временных Рядов

- Данные о потерях и предсказаниях загружаются и обрабатываются.
- Применяется скользящее среднее (сглаживание) для определения вероятности наличия аномалий.
- Сверяются временные ряды для предсказаний и данных, при необходимости происходит дополнительная синхронизация временных меток.

### Определение Интервалов Аномалий

- Используя настроенные пороги и параметры, алгоритм определяет интервалы, где предсказанные значения выходят за пределы нормы.
- Для каждого интервала вычисляется среднее значение потерь и определяются основные датчики, ответственные за аномалии.

### Сохранение Результатов

- Результаты анализа сохраняются в JSON файлы, организованные по названиям станций.

### Постобработка

- Для каждой обработанной группы данных проверяется, входит ли она в список известных интервалов с аномалиями.
- Если да, для каждого такого интервала определяется вхождение в выявленные аномалии, что позволяет оценить качество анализа через пересечение по объединению (IoU).

### Дополнительные Особенности

- В коде предусмотрены механизмы обработки ошибок и несоответствий в данных, такие как отсутствие синхронизации между временными рядами и различия в длине данных.
- Код динамически создает директории и обрабатывает данные в батчах, что облегчает масштабирование проекта.
