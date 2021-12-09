import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from preprocess.text_process import CorrectSpelling
from transform.text_transforms import (TextUnionTransform,
                                       SimpleTokenizerTransformer,
                                       CorrectSpellingTransformer,
                                       MystemLemmatizerTransformer,
                                       KeepCommonTransformer,
                                      )

def TfIdfColumnTransformer(columns=["name_dish", "product_description"],
                           remainder="passthrough"):
    return ColumnTransformer([
            ("text_tf_idf",
             Pipeline([
                (  "text_union",
                    TextUnionTransform(columns),
                ),
                (  "count", 
                    CountVectorizer()
                ),
                (  "tf_idf",
                    TfidfTransformer(),
                ),
             ]),
             columns,
            ),
        ],
        remainder=remainder,
        )


class PriceOutliersTransformer(BaseEstimator, TransformerMixin):
    def price_filter(price):
        if price > 100000:
            price /= 1000
        if price > 35000:
            price /= 100
        return int(price)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.apply(lambda col: col.apply(PriceOutliersTransformer.price_filter))


class TransformerToDF:
    def __init__(self, transformer, columns):
        self.transformer = transformer
        self.columns = columns

    def fit(self,  *args, **kwargs):
        return self.transformer.fit(*args, **kwargs)

    def fit_transform(self,  *args, **kwargs):
        return pd.DataFrame(self.transformer.fit_transform(*args, **kwargs),
                            columns=self.columns)

    def transform(self,  *args, **kwargs):
        return pd.DataFrame(self.transformer.transform(*args, **kwargs),
                            columns=self.columns)


def PreprocessNameDescPrice(to_spell_path, max_len=10000, threshold_cnt=10, unk_token=' ') -> ColumnTransformer:

    word_to_correct = np.load(to_spell_path, allow_pickle=True).item()
    corrector = CorrectSpelling(word_to_correct)


    transformer = ColumnTransformer(
        [
            (  "text_process_pipeline",
                Pipeline([
                    (  "tokenizer",
                        SimpleTokenizerTransformer(),
                    ),
                    (  "corrector",
                        CorrectSpellingTransformer(corrector),
                    ),
                    (  "lemmatizer",
                        MystemLemmatizerTransformer(),
                    ),
                    (  "keep_common",
                        KeepCommonTransformer(max_len, threshold_cnt, unk_token),
                    ),
                ]),
                ["name_dish", "product_description"],
            ),

            (  "price_pipeline",
                Pipeline([
                    (  "filter_outliers",
                        PriceOutliersTransformer(),
                    ),
                    (  "normalization",
                        MinMaxScaler(),
                    ),
                ]),
                ["price"],
            ),
        ],
        remainder="drop",
    )

    return TransformerToDF(transformer, ["name_dish", "product_description", "price"])


def PreprocessNameDesc(to_spell_path, max_len=10000, threshold_cnt=10, unk_token=' ') -> ColumnTransformer:

    word_to_correct = np.load(to_spell_path, allow_pickle=True).item()
    corrector = CorrectSpelling(word_to_correct)


    transformer = ColumnTransformer(
        [
            (  "text_process_pipeline",
                Pipeline([
                    (  "tokenizer",
                        SimpleTokenizerTransformer(),
                    ),
                    (  "corrector",
                        CorrectSpellingTransformer(corrector),
                    ),
                    (  "lemmatizer",
                        MystemLemmatizerTransformer(),
                    ),
                    (  "keep_common",
                        KeepCommonTransformer(max_len, threshold_cnt, unk_token),
                    ),
                ]),
                ["name_dish", "product_description"],
            ),
        ],
        remainder="drop",
    )

    return TransformerToDF(transformer, ["name_dish", "product_description"])

