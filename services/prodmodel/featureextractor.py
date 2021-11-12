import pickle
import numpy as np
import torch
from utils import load_files_from_s3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor:

    def __init__(self, path):
        self.path = path
        self.word_to_idx = self.__get_word_to_idx()


    def __get_word_to_idx(self):
        with open(self.path + '/dataformodel/word_to_idx.pkl', 'rb') as w_t_i_file:
            return pickle.load(w_t_i_file) 

    def get_features(self, data):
        data['X'] = data['name_dish'] + ' ' + data['product_description']
        data['X_len'] = data['X'].apply(lambda x: len([w for w in x.split(' ') if w != '']))
        data['X_idx'] = data['X'].apply(lambda x: [self.word_to_idx[word] for word in x.split(' ') if word != ''])

        return [torch.LongTensor(data['X_idx']).to(device), torch.LongTensor([data['X_len'].values]).to(device)]

        
def get_feature_extractor(path):
    load_files_from_s3('models/baseline/word_to_idx.pkl', path, '/dataformodel/word_to_idx.pkl')
    feature_extractor = FeatureExtractor(path)

    return feature_extractor