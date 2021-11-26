
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocess.transform_pipeline import TfIdfColumnTransformer


def TfIdf_Price_LogReg(C=1) -> Pipeline:
    return Pipeline([
        (  "tf_idf_price",
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
