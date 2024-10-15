import os
from loguru import logger

def count_files(directory):
    """Подсчитывает количество файлов в указанной директории."""
    # Инициализируем счётчик файлов
    file_count = 0

    # Проходим по всем элементам в директории
    for entry in os.listdir(directory):
        # Полный путь к элементу
        full_path = os.path.join(directory, entry)
        # Если элемент является файлом, увеличиваем счётчик
        if os.path.isfile(full_path):
            file_count += 1

    return file_count


def extract_group_ids(station_data):
    group_ids = []
    # Проверяем, существует ли ключ station_name в данных
    # Проходим по каждой группе и собираем ID
    for group in station_data['groups']:
        group_ids.append(group['id'])
    return group_ids

import pandas as pd

def find_indices_in_range(df, start_date, end_date):
    """
    Возвращает индексы из DataFrame, которые попадают в заданный временной промежуток.

    Параметры:
    df : pandas.DataFrame
        DataFrame с индексами в формате datetime.
    start_date : str
        Строка с начальной датой промежутка в формате "YYYY-MM-DD HH:MM:SS".
    end_date : str
        Строка с конечной датой промежутка в формате "YYYY-MM-DD HH:MM:SS".

    Возвращает:
    pandas.Index
        Индексы, которые попадают в указанный временной промежуток.
    """
    
    # Конвертируем строки в datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    

    # Фильтруем DataFrame для получения индексов в заданном промежутке
    filtered_df = df.loc[start_date:end_date]

    # Возвращаем индексы отфильтрованного DataFrame
    return (filtered_df['Row_Number'][0], filtered_df['Row_Number'][-1])
def calculate_iou(true_interval, predicted_interval):
    # Распаковка интервалов
    true_start, true_end = true_interval
    pred_start, pred_end = predicted_interval

    # Вычисление пересечения
    intersection_start = max(true_start, pred_start)
    intersection_end = min(true_end, pred_end)
    intersection_length = max(0, intersection_end - intersection_start)

    # Вычисление объединения
    union_start = min(true_start, pred_start)
    union_end = max(true_end, pred_end)
    union_length = union_end - union_start

    # Вычисление IoU
    iou = intersection_length / union_length if union_length > 0 else 0
    return iou



def union_intervals(documents):
    """
    Объединяет список интервалов в один или возвращает [0, 0], если список пуст.

    Parameters:
        documents (list): список словарей, где каждый словарь содержит ключ 'index' с парой значений (начало и конец интервала).

    Returns:
        list: список из двух элементов, представляющих начальную и конечную точки объединенного интервала.
    """
    if not documents:
        return [0, 0]

    # Инициализируем начальные значения первым документом
    start, end = documents[0]['index']
    logger.info(f"Find interval {documents[0]['index']}")

    # Перебираем оставшиеся документы, начиная со второго
    for doc in documents[1:]:
        logger.info(f"Find interval {doc['index']}")
        _, current_end = doc['index']
        end = current_end  # Обновляем конец интервала

    logger.info(f"Union interval [{start}, {end}]")
    return [start, end]