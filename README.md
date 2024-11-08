# Исследования по поиску методов детектирования аномалий

Данный репозиторий содержит методы для выявления аномалий.

Реализован методы ECOD,CBLOF и ALAD из фреймворка PyOD.
Также реализованы методы PcaAD и PcaReconstryctionError из фреймворка ADTK.
Стоит отметить что также были добавлены функции в сами модели для получения loss по каждой фиче (датчику) отдельно.
## Структура [проекта](ML_methods/):
  Код каждой модели находится в отдельной папке. В папке содержится 2 файла: (такая структура во всех моделях кроме ECOD)
      1) Файл с основным кодом (название метода.py)
      2) Файл для отладки кода на одной группе (One_Group_Test_метод.py)

## [Метод ECOD](ML_methods/ECOD/).
Основной скрипт [ECOD_Gpt](ML_methods/ECOD/GPT_ECOD.py)
В файле [utils_def.py](utils/utils_def.py) функция dir_maker создает диррикторию для отчета по нужному датасету.
После создания всех необходимых файлом можно запускать скрипт run для формирования отчета analysis_report.txt

## [Метод ALAD](ML_methods/ALAD/).
Основной скрипт [ALAD.py](ML_methods/ALAD/ALAD.py)

## [Метод CBLOF](ML_methods/CBLOF/).
Основной скрипт [CBLOF.py](ML_methods/CBLOF/CBLOF.py)

## [Метод PCA](ML_methods/PCA/).
Основной скрипт [PCA.py](ML_methods/PCA/PCA_anomaly.py)

Кроме модели PcaAD так же реализован поиск аномалий с помощью трансформера PcaReconstructionError
## [Метод PcaReconstructionError](ML_methods/PcaReconstructionError/).
Основной скрипт [PcaReconstructionError.py](ML_methods/PcaReconstructionError/PcaReconstructionError_anomaly.py)

## Что не реализованно пока что:
1) Далее будут добавляться новые методы для выявления аномалий (на очереди PcaRegressor)
2) Будет автоматизировано создание дерикторий и запуск всех необходимиых файлов (формирование дерикторий-> запуск [ECOD_Gpt](ML_methods/ECOD/GPT_ECOD.py)-> запуск run скрипта-> формирования отчета analysis_report.txt)

