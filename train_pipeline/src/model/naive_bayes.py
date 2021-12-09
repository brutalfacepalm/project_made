from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB

from preprocess.transform_pipeline import TfIdfColumnTransformer


def TfIdf_ComplementNB() -> Pipeline:
    return Pipeline([
        (  "tf_idf_price",
           TfIdfColumnTransformer(),
        ),

        (  "naive_bayes",
            ComplementNB(),
        ),
    ])
