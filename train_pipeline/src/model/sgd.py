import numpy as np
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from preprocess.transform_pipeline import TfIdfColumnTransformer


def TfIdf_SGD(alpha=0.0001) -> Pipeline:
    return Pipeline([
        (  "tf_idf",
           TfIdfColumnTransformer(),
        ),

        (  "sgd",
            SGDClassifier(loss='log',
                          alpha=alpha,
                          class_weight="balanced",
                          max_iter=100,
                          ),
        ),
    ])


class WeightSGD(SGDClassifier):
    def __init__(self, weight_adding=0, **kwargs):
        super().__init__(**kwargs)
        self.weight_adding = weight_adding

    def fit(self, X, y, **kwargs):
        weights = len(y) / len(np.unique(y)) / np.bincount(y) + self.weight_adding
        self.class_weight = {i : w for i, w in enumerate(weights)}
        super().fit(X, y, **kwargs)


def WeightAdding_SGD(alpha=0.0001, weight_adding=0) -> Pipeline:
    return Pipeline([
        (  "tf_idf",
           TfIdfColumnTransformer(),
        ),

        (  "sgd",
            WeightSGD(loss='log',
                          alpha=alpha,
                          #class_weight="balanced",
                          max_iter=1000,
                          weight_adding=weight_adding,
                          ),
        ),
    ])

