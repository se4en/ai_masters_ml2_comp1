#### Отчет по [соревнованию предсказания оценки стоимости недвижимости](https://www.kaggle.com/competitions/aim2023-guess-the-price/overview)

## Инструкция по запуску

1. ```pip install -r requirements.txt```

2. ```python src/train.py -m```

После этого запустится процесс подбора гиперпараметров с помощью optuna, результаты которого можно будет наблюдать через UI mlflow, который будет доступен на  после команды:

3. ```mlflow ui --port 5050```

## Описание решения

 - #### Выбор модели
 
    Т.к. данные в задаче были табличными, то в качестве моеделей было решено взять бустинги. Вначале пробовал LGBM, потом пересел на CatBoost, стало получше. При подборе гиперпараметров оптимизировал число деревьев, глубину, lr, к-т регуляризации (конфиги со значениями гиперпараметров лежат [здесь](src/conf/search_space/)). В соревновании использовалась метрика RMSLE, но при обучении моделей метрику менять не пробовал.
 
 - #### Предобработка данных
    
    посмотрел на все признаки и выделил подозрительные значения, соответсвующие объекты не удалял, заменил признаки на nan’ы.
 
 - #### Валидация
 
    Т.к. для данных в задаче была явно указана временная шкала, и в тестовые данных находились объекты с более поздним значением времени, то было решено использовать аналогичную схему валидации. В файле  реализованы две схемы валидации -- с увеличивающимся . Размер val set старался устанавливать таким, чтобы сохранялось отношения обучющих данным к тестовым. В целом в плане корреляции значения метрики на валидации и public score разницу между двумя вышеописанными подходами не заметил, но все равно картина была лучше, чем при использовании обычной кросс-валидации. 
 
 - #### Формирование финального сабмита
    
    Для формирования финального сабмита я взял три лучших по public score сабмита и усреднил их, за счет чего удалось немного улучшить значение public score, и в чуть большей степени улучшить значение private score.

## Описание пайплайна

Для проведения и управлением экспериментами использовалась связка из  
