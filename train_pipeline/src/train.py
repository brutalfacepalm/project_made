import logging
from typing import List, Optional

from timebudget import timebudget
import hydra
from omegaconf import DictConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


from utils import load_data, split_data, reduce_data, extract_target
from preprocess.text_process import word_counter


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # загружаем данные без дубликатов
    data = load_data(config.load_data.in_path, config.load_data.out_path)
    logger.debug(f'load data, len={len(data)}, columns={data.columns}')

    # отделяем тестовый набор
    train, test = split_data(data, config.test_split.test_size)
    logger.debug(f'split data, train={len(train)}, test={len(test)}')
    #logger.debug(f'{train.tags_menu.value_counts()}')
    #logger.debug(f'{test.tags_menu.value_counts() / train.tags_menu.value_counts()}')

    # уменьшаем количество тренировочных данных, если надо
    train = reduce_data(train, config.train_reduce.class_max_len)
    logger.debug(f'reduce train size={len(train)}')
    #logger.debug(f'{train.tags_menu.value_counts()}')

    
    X, y = extract_target(train)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=config.valid_split.val_size,
                                                          random_state=0)
    logger.debug(f'split data, train={len(X_train)}, valid={len(X_valid)}')

    # предобработка данных   
    preprosess = hydra.utils.instantiate(config.preprocess)
    with timebudget("preprocess transform"):
        X_train = pd.DataFrame(preprosess.fit_transform(X_train), 
                               columns=X_train.columns)

    # словарь обучения
    name_words = word_counter(X_train['name_dish'])
    desc_words = word_counter(X_train['product_description'])

    logger.debug(f'train name_dish words: {len(name_words)}')
    logger.debug(f'train product_description words: {len(desc_words)}')
    logger.debug(f'train total words: {len(name_words + desc_words)}')

    # модель
    model = hydra.utils.instantiate(config.model)
    model.fit(X_train, y_train)

    y_pred = model.predict(preprosess.transform(X_valid))

    report = pd.DataFrame(classification_report(y_valid, 
                                                y_pred, 
                                                output_dict=True, 
                                                target_names=target_encoder.classes_)
                          ).T
    print(report)
    # import pdb
    # pdb.set_trace()
    


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()
