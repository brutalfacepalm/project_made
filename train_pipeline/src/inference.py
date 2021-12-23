import pickle
import pandas as pd

class InferenceModel:
    def __init__(self, model, preprocess, target_encoder):
        self.model = model
        self.preprocess = preprocess
        self.target_encoder = target_encoder

    def predict_proba(self, X):
        X = self.preprocess.transform(X)
        pred_proba = self.model.predict_proba(X)
        return pd.DataFrame(pred_proba, columns=self.target_encoder.classes_)

    def predict(self, X):
        X = self.preprocess.transform(X)
        pred = self.model.predict(X)
        return self.target_encoder.inverse_transform(pred)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)


if __name__ == '__main__':
    model = InferenceModel.load('model.pkl')
    query = pd.DataFrame.from_records([
        {'name_dish' : 'оливье', 'product_description' : 'яйцо, майонер, колбаса, горошек'}
    ])
    tag = model.predict(query)
    print(query)
    print(tag[0])
    tag_prob = model.predict_proba(query)
    print(tag_prob.iloc[0])


