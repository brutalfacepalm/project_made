import pickle
import numpy as np
import torch
from transformers import BertTokenizer
from utils import load_files_from_s3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence', do_lower_case=True,
                                                  additional_special_tokens=['[NAME]', '[DESC]', '[PRICE]'])
        self.max_len = 85

    def get_features(self, data):
        input_ids = []
        attention_masks = []

        encoded_dict = self.tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

def get_feature_extractor():
    feature_extractor = FeatureExtractor()

    return feature_extractor