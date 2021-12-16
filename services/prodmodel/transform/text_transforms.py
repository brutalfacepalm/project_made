from collections import Counter

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from preprocess.text_process import (SimpleTokenizer, 
                                     CorrectSpelling, 
                                     MystemLemmatizer,
                                     word_counter,
                                     Mystem)


class TextUnionTransform(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        out = X[self.cols[0]]
        for col in self.cols:
            out += ' ' + X[col]
        return out

        # X = pd.DataFrame(X)
        # out = X.iloc[:, self.i_cols[0]]
        # for i_col in self.i_cols[1:]:
        #     out += ' ' + X.iloc[:, i_col]
        # return out


class SimpleTokenizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or SimpleTokenizer()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.apply(lambda col: col.apply(self.tokenizer.tokenize))


class CorrectSpellingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, corrector: CorrectSpelling):
        self.corrector = corrector

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.apply(lambda col: col.apply(self.corrector.correct))


class MystemLemmatizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatizer=None):
        self.lemmatizer = lemmatizer or MystemLemmatizer()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        mystem = Mystem()
        to_lemmas = lambda word : self.lemmatizer.lemmatize_sentence(word, mystem)
        return X.apply(lambda col: col.apply(to_lemmas))


class KeepCommonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_len=10000, threshold_cnt=10, unk_token=' '):
        self.max_len = max_len
        self.threshold_cnt = threshold_cnt
        self.unk_token = unk_token
        self.keep_words = set()

    def class_keep_words(self, X: pd.DataFrame):
        min_cnt = int(min(self.threshold_cnt, 0.01 * len(X)))
        word_counters = [
            word_counter(X[column])
            for column in X
        ]
        keep_counters = [
            Counter({
                w : c 
                for w, c in cnt.most_common(self.max_len)
                if c >= min_cnt
            })
            for cnt in word_counters
        ]
        keep_counter = sum(keep_counters, Counter())
        keep_words = set(w for w, c in keep_counter.items())
        return keep_words

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if not y:
            self.keep_words = self.class_keep_words(X)
        else:
            keep_words = set()
            for class_name in y.unique():
                keep_words.update(
                    self.class_keep_words(X[y == class_name])
                    )
            self.keep_words = keep_words

        return self

    def keep_common(self, sentence):
        if not sentence:
            return sentence
        return ' '.join([
            (word 
            if word in self.keep_words 
            else self.unk_token
            )
            for word in sentence.split()
        ])

    def transform(self, X: pd.DataFrame):
        return X.apply(lambda col: col.apply(self.keep_common))