from pickle import load as load_pickle

import torch
from torch import load as load_torch
from torch import nn, device, cat, FloatTensor, cuda
from utils import load_files_from_s3
from transformers import BertForSequenceClassification, BertConfig

device = device('cuda' if cuda.is_available() else 'cpu')

NUM_LABELS = 39


def get_model(path):
    path_model = path + '/dataformodel/modelbert'
    model = torch.load(path_model, map_location=device)

    return model


