import os
import logging
from typing import List, Optional

from timebudget import timebudget
import hydra
from omegaconf import DictConfig, OmegaConf

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import mlflow


from utils import load_data, load_test_data, reduce_Xy, extract_target, write_report, check_precision_recall
from inference import InferenceModel
from preprocess.text_process import word_counter

import torch
import numpy as np
import random
import pickle


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def preprocess_data(config):

        # загружаем данные без дубликатов
        data = load_data(config.load_data.in_path, config.load_data.out_path)
        logger.debug(f'load data, len={len(data)}, columns={data.columns}')

        # train, test = split_data(data, config.test_split.test_size)
        # logger.debug(f'split data, train={len(train)}, test={len(test)}')
        #logger.debug(f'{train.tags_menu.value_counts()}')
        #logger.debug(f'{test.tags_menu.value_counts() / train.tags_menu.value_counts()}')
       
        X, y = extract_target(data)
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                            stratify=y,
                                                            test_size=config.valid_split.val_size,
                                                            random_state=0)
        # предобработка данных
        preprosess = hydra.utils.instantiate(config.preprocess)
        with timebudget("preprocess transform"):
            X_train = preprosess.fit_transform(X_train)
            X_valid = preprosess.transform(X_valid)

        # X_train = X_train.drop_duplicates()
        # y_train = y_train[X_train.index]

        # X_valid = X_valid.drop_duplicates()
        # y_valid = y_valid[X_valid.index]

        # preprosess_train = X_train.copy()
        # preprosess_train['target'] = target_encoder.inverse_transform(y_train)
        # preprosess_train = preprosess_train.drop_duplicates()
        # preprosess_train.to_csv('preprosess_train.csv', index=False)

        # preprosess_valid = X_valid.copy()
        # preprosess_valid['target'] = target_encoder.inverse_transform(y_valid)
        # preprosess_valid = preprosess_valid.drop_duplicates()
        # preprosess_valid.to_csv('preprosess_valid.csv', index=False)
        # print(f'drop_duplicates, train={len(preprosess_train)}, valid={len(preprosess_valid)}')
        return X_train, y_train, X_valid, y_valid, target_encoder, preprosess


def cache_preprocess_data(config, cache_path='/tmp/preprocess.pkl'):
    if not config.preprocess_cache or not os.path.exists(cache_path):
        preprocess_out = preprocess_data(config)
        with open(cache_path, 'wb') as f:
            pickle.dump(preprocess_out, f)
    else:
        with open(cache_path, 'rb') as file:
            preprocess_out = pickle.load(file)
    return preprocess_out


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # import json
    # print(json.dumps(OmegaConf.to_container(config), indent=4, sort_keys=False))

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.debug(f'experiment_name: {config.mlflow.experiment_name}')
    logger.debug(f'mlflow_uri: {config.mlflow.tracking_uri}')

    # os.environ["AWS_ACCESS_KEY_ID"] = "..."
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.us-east-2.amazonaws.com/" 

    with mlflow.start_run():

        X_train, y_train, X_valid, y_valid, target_encoder, preprosess \
            = cache_preprocess_data(config)
        logger.debug(f'preprocess data: train={len(X_train)}, valid={len(X_valid)}')

        # уменьшаем количество тренировочных данных, если надо
        # X_train, y_train = reduce_Xy(X_train, y_train, config.train_reduce.class_max_len)
        # logger.debug(f'reduce train size={len(X_train)}')
        # #logger.debug(f'{train.tags_menu.value_counts()}')

        # словарь обучения
        name_words = word_counter(X_train['name_dish'])
        desc_words = word_counter(X_train['product_description'])

        logger.debug(f'train name_dish words: {len(name_words)}')
        logger.debug(f'train product_description words: {len(desc_words)}')
        logger.debug(f'train total words: {len(name_words + desc_words)}')


        # модель
        model = hydra.utils.instantiate(config.model)

        try:
            model.fit(X_train, y_train, X_valid, y_valid, target_encoder)
        except:
            model.fit(X_train, y_train)

        # оцениваем на train
        pred_train = model.predict(X_train)
        train_report = pd.DataFrame(metrics.classification_report(y_train, 
                                                        pred_train, 
                                                        output_dict=True, 
                                                        target_names=target_encoder.classes_)
                                    ).T
        print('train report:')
        print(train_report)

        # оцениваем на валидации
        pred_valid = model.predict(X_valid)
        valid_report = pd.DataFrame(metrics.classification_report(y_valid, 
                                                        pred_valid, 
                                                        output_dict=True, 
                                                        target_names=target_encoder.classes_)
                                    ).T
        print('validation report:')
        print(valid_report)

        # оцениваем на тестовом датасете
        X_test, y_test = load_test_data(config.test_data_path)
        logger.debug(f'test data len = {len(X_test)}, columns = {X_test.columns}')

        X_test = preprosess.transform(X_test)
        y_test = target_encoder.transform(y_test)
        pred_test = model.predict(X_test)
        test_report = pd.DataFrame(metrics.classification_report(y_test, 
                                                        pred_test, 
                                                        output_dict=True, 
                                                        target_names=target_encoder.classes_)
                                    ).T
        print('test report:')
        print(test_report)

        #check_precision_recall(y_test, pred_test, n_classes=len(target_encoder.classes_))

        inference_model = InferenceModel(model, preprosess, target_encoder)
        inference_model.save(config.inference_path)

        mlflow.log_metrics({
            'train_precision' : train_report['precision']['macro avg'],
            'train_recall' : train_report['recall']['macro avg'],
            'valid_precision' : valid_report['precision']['macro avg'],
            'valid_recall' : valid_report['recall']['macro avg'],
            'test_precision' : test_report['precision']['macro avg'],
            'test_recall' : test_report['recall']['macro avg'],
            })
        mlflow.log_params(config.model)

        write_report(train_report,  'train_report.html')
        write_report(valid_report, 'valid_report.html')
        write_report(test_report,  'test_report.html')

        mlflow.log_artifacts(os.getcwd())


    # import pdb
    # pdb.set_trace()
    


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()
