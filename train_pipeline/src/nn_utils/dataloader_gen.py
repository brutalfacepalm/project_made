import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import WeightedRandomSampler

class FullDataset(Dataset):
    def __init__(self, name_idxs, desc_idxs, price, y_idx=None):
        self.name_idxs = name_idxs
        self.name_len = list(map(len, name_idxs))
        self.desc_idxs = desc_idxs
        self.desc_len = list(map(len, desc_idxs))
        self.union_idxs = name_idxs + desc_idxs
        self.union_len = self.name_len + self.desc_len
        self.price = price
        self.y_idx = y_idx

    def __len__(self):
        return len(self.price)
    
    def __getitem__(self, idx):       
        out = (self.name_idxs[idx], 
               self.name_len[idx],
               self.desc_idxs[idx],
               self.desc_len[idx],
               self.union_idxs[idx],
               self.union_len[idx],
               self.price[idx],

        )
        if self.y_idx is not None:
            out = *out, self.y_idx[idx]
        return out


def collate_fn(batch, pad_idx):
    if len(batch[0]) == 8: 
        name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len, price, y_idx = zip(*batch)
    else:
        name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len, price = zip(*batch)
        y_idx = None

    name_len = torch.LongTensor(name_len)
    desc_len = torch.LongTensor(desc_len)
    union_len = torch.LongTensor(union_len)

    name_idxs = pad_sequence([torch.LongTensor(idxs) for idxs in name_idxs],
                             padding_value=pad_idx, 
                             batch_first=True)
    desc_idxs = pad_sequence([torch.LongTensor(idxs) for idxs in desc_idxs],
                             padding_value=pad_idx, 
                             batch_first=True)
    union_idxs = pad_sequence([torch.LongTensor(idxs) for idxs in union_idxs],
                             padding_value=pad_idx, 
                             batch_first=True)
    price = torch.FloatTensor(price)

    out = name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len, price
    if y_idx is not None:
        out = *out, torch.LongTensor(y_idx)
    return out



def dataloader_gen(X, y, text_to_idxs_fun, pad_idx, batch_size=64, shuffle=True):
    name_idxs = X['name_dish'].apply(text_to_idxs_fun).values
    desc_idxs = X['product_description'].apply(text_to_idxs_fun).values
    price = X['price'].values
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
    price = X['price'].values
    y_idx = y

    dataset = FullDataset(name_idxs, desc_idxs, price, y_idx)
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=WeightedRandomSampler(weights, len(weights)),
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    return dataloader
