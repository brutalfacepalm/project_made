import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess.text_process import CorrectSpelling
from transform.text_transforms import (SimpleTokenizerTransformer,
                                       CorrectSpellingTransformer,
                                       MystemLemmatizerTransformer,
                                       KeepCommonTransformer,
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



def PreprocessTransformer(to_spell_path, max_len=10000, threshold_cnt=10, unk_token=' ') -> ColumnTransformer:

    word_to_correct = np.load(to_spell_path, allow_pickle=True).item()
    corrector = CorrectSpelling(word_to_correct)


    return ColumnTransformer(
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
                    (  "standard_scaler",
                        StandardScaler(),
                    ),
                ]),
                ["price"],
            ),
        ],
        remainder="passthrough",
    )

