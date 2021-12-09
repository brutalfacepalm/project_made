import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import WeightedRandomSampler


class FullDataset(Dataset):
    def __init__(self, name_idxs, desc_idxs, price=None, y_idx=None):
        self.name_idxs = name_idxs
        self.name_len = list(map(len, name_idxs))
        self.desc_idxs = desc_idxs
        self.desc_len = list(map(len, desc_idxs))
        self.union_idxs = name_idxs + desc_idxs
        self.union_len = self.name_len + self.desc_len
        self.price = price
        self.y_idx = y_idx

    def __len__(self):
        return len(self.name_idxs)
    
    def __getitem__(self, idx):       
        out = (self.name_idxs[idx], 
               self.name_len[idx],
               self.desc_idxs[idx],
               self.desc_len[idx],
               self.union_idxs[idx],
               self.union_len[idx],
        )
        if self.price is not None:
            out = *out, self.price[idx]
        if self.y_idx is not None:
            out = *out, self.y_idx[idx]
        return out


def collate_fn(batch, pad_idx):
    inputs = zip(*batch)
    outputs = []
    for input in inputs:
        if isinstance(input[0], int) or isinstance(input[0], np.int64):
            output = torch.LongTensor(np.array(input))
        elif isinstance(input[0], list):
            output = pad_sequence([torch.LongTensor(np.array(idxs))
                                   for idxs in input
                                   ],
                                   padding_value=pad_idx, 
                                   batch_first=True
                                )
        elif isinstance(input[0], float):
            output = torch.FloatTensor(np.array(input))
        else:
            # import pdb
            # pdb.set_trace()
            raise Exception('input type=', type(input[0]))
        outputs.append(output)
    return outputs


def dataloader_gen(X, y, text_to_idxs_fun, pad_idx, batch_size=64, shuffle=True):
    name_idxs = X['name_dish'].apply(text_to_idxs_fun).values
    desc_idxs = X['product_description'].apply(text_to_idxs_fun).values
    price = X['price'].values if 'price' in X.columns else None
    y_idx = y
    
    dataset = FullDataset(name_idxs, desc_idxs, price, y_idx)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    return dataloader


def weighted_dataloader_gen(X, y, text_to_idxs_fun, pad_idx, weights, batch_size=64):
    name_idxs = X['name_dish'].apply(text_to_idxs_fun).values
    desc_idxs = X['product_description'].apply(text_to_idxs_fun).values
    price = X['price'].values if 'price' in X.columns else None
    y_idx = y

    dataset = FullDataset(name_idxs, desc_idxs, price, y_idx)
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=WeightedRandomSampler(weights, len(weights)),
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    return dataloader
