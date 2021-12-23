
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocess.transform_pipeline import TfIdfColumnTransformer


def TfIdf_LogReg(C=1) -> Pipeline:
    return Pipeline([
        (  "tf_idf",
           TfIdfColumnTransformer(),
        ),

        (  "log_reg",
            LogisticRegression(C=C,
                                multi_class='multinomial', 
                                solver='lbfgs', 
                                class_weight="balanced",
                                ),
        ),
    ])


class WeightLogReg(LogisticRegression):
    def __init__(self, weight_adding=0, **kwargs):
        super().__init__(**kwargs)
        self.weight_adding = weight_adding

    def fit(self, X, y, **kwargs):
        weights = len(y) / len(np.unique(y)) / np.bincount(y) + self.weight_adding
        self.class_weight = {i : w for i, w in enumerate(weights)}
        super().fit(X, y, **kwargs)


def WeightAdding_LogReg(C=1, weight_adding=0) -> Pipeline:
    return Pipeline([
        (  "tf_idf",
           TfIdfColumnTransformer(),
        ),

        (  "log_reg",
            WeightLogReg(C=C,
                        multi_class='multinomial', 
                        solver='lbfgs', 
                        # class_weight="balanced",
                        weight_adding=weight_adding,
                        ),
        ),
    ])




