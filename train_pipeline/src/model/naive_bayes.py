from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB

from transform.text_transforms import TextUnionTransform

def TfidfComplementNB() -> Pipeline:
    return Pipeline([
        (  "text_union",
            TextUnionTransform([0, 1]),
        ),
        (  "count", 
            CountVectorizer()
        ),
        (  "tf_idf",
            TfidfTransformer(),
        ),
        (  "naive_bayes",
            ComplementNB(),
        ),
    ])



