import pickle
from scipy.special import softmax
import numpy as np


class InferenceModel:
    def __init__(self, model, preprocess, target_encoder):
        self.model = model
        self.preprocess = preprocess
        self.target_encoder = target_encoder

    def predict(self, X):
        X = self.preprocess.transform(X)
        log_prediction = self.model.predict_log_proba(X)
        predictions = softmax(log_prediction)[0]
        precent_predictions = (predictions - np.mean(predictions))/max(predictions)
        # precent_predictions = predictions
        label_pred = np.argmax(precent_predictions).reshape(1, -1)
        category = self.target_encoder.inverse_transform(label_pred)[0]
        return category, label_pred, precent_predictions.reshape(1, -1)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)