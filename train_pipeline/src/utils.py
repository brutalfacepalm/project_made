from typing import Tuple

import logging
import pickle
import os

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_test_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    #logger.debug(f"load test data from {path}")
    return data.drop(columns='target'), data['target']


def load_data(in_path: str, out_path: str) -> pd.DataFrame:
    if not os.path.exists(out_path):
        with open(in_path, 'rb') as file:
            data = pickle.load(file)
        logger.debug(f"load data from {in_path}")

        data['tags_menu'] = data['tags_menu'].apply(lambda x: [k for k in x.keys()][0])
        data.dropna(inplace = True)
        data.drop_duplicates(subset = ['name_dish','product_description','tags_menu'], inplace = True)

        data = data[['name_dish','product_description','price', 'tags_menu']].reset_index(drop=True)

        data.to_csv(out_path, index=False)
        logger.debug(f"write data to {out_path}")
    else:
        data = pd.read_csv(out_path)
        logger.debug(f"read data from {out_path}")
        #logger.debug(data.head())
    return data


def split_data(data: pd.DataFrame, test_size: float, seed=0) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train, test = train_test_split(data, 
                                   stratify=data['tags_menu'], 
                                   test_size=test_size, 
                                   random_state=seed)
    return train, test


def sample_class(data: pd.DataFrame, class_value: str, n_cases: int, seed=0) -> pd.DataFrame:
    class_data = data[data['tags_menu'] == class_value]
    if len(class_data) <= n_cases:
        return class_data
    else:
        return class_data.sample(n=n_cases, random_state=seed)


def extract_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns='tags_menu'), data['tags_menu']


def reduce_data(data: pd.DataFrame, class_max_len: int) -> pd.DataFrame:
    data_reduced = pd.concat(
                    [sample_class(data, name, class_max_len, seed=i)
                    for i, name in enumerate(data['tags_menu'].unique())],
                    axis=0).reset_index(drop=True)
    return data_reduced


def reduce_Xy(X: pd.DataFrame, y:pd.Series, class_max_len: int) -> pd.DataFrame:
    data = X.copy()
    data['tags_menu'] = y
    data_reduced = reduce_data(data, class_max_len)
    return extract_target(data_reduced)


def write_report(report, path):
    html = report.to_html()
    with open(path, "w", encoding="utf-8") as file:
        file.writelines('<meta charset="UTF-8">\n')
        file.write(html)


def check_precision_recall(y_true, y_pred, n_classes):
    from sklearn import metrics
    import numpy as np

    sk_precision = metrics.precision_score(y_true, y_pred, average='macro')
    sk_recall = metrics.recall_score(y_true, y_pred, average='macro')
    print(f'sklearn precision: {sk_precision}, recall: {sk_recall}')

    target_true = np.zeros(n_classes)
    predict_true = np.zeros(n_classes)
    correct_true = np.zeros(n_classes)
    recall_classes = np.zeros(n_classes)
    precision_classes = np.zeros(n_classes)
    for i in range(n_classes):
        target_true[i] = np.sum((y_true == i).astype(int))
        predict_true[i] = np.sum((y_pred == i).astype(int))
        correct_true[i] = np.sum((y_true == i).astype(int) * (y_pred == i).astype(int))
        recall_classes[i] = correct_true[i] / target_true[i] if target_true[i] != 0 else 0
        precision_classes[i] = correct_true[i] / predict_true[i] if predict_true[i] != 0 else 0
    print(f'checked precision: {np.mean(precision_classes)}, recall: {np.mean(recall_classes)}')


