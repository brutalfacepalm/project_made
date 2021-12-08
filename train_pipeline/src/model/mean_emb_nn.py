import numpy as np
import torch
from torch import nn


class MeanEmbeddingModel(nn.Module):
    def __init__(self):
        super(MeanEmbeddingModel, self).__init__()

    def set(self, word_embeddings, out_dim):
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                      freeze=True)
        
        self.fc = nn.Linear(self.embedding.weight.shape[-1] * 2 + 1, out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        return self

    def _calc_emb(self, text_idxs, text_len):
        emb = self.embedding(text_idxs) # batch_len, seq_len, emb_shape
        emb = torch.sum(emb, dim = 1) # batch_len, emb_shape
        text_len[text_len == 0] = 1
        emb = torch.div(emb, text_len.unsqueeze(1)) # batch_len, emb_shape
        return emb

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len, price):
        name_emb = self._calc_emb(name_idxs, name_len) # batch_len, emb_shape
        desc_emb = self._calc_emb(desc_idxs, desc_len) # batch_len, emb_shape
        input = torch.cat((name_emb, desc_emb, price.unsqueeze(1)), dim = -1) #batch_len, 2 * emb_shape + 1
        
        input = self.fc(input) # batch_len, out_dim
        return input


class CatMeanEmbeddingModel(MeanEmbeddingModel):
    def __init__(self):
        super(CatMeanEmbeddingModel, self).__init__()
    
    def set(self, word_embeddings, out_dim):
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                      freeze=True)
        self.fc = nn.Linear(self.embedding.weight.shape[-1] + 1, out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len, price):
        emb = self._calc_emb(torch.cat((name_idxs, desc_idxs), dim = -1),
                             name_len + desc_len) # batch_len, emb_shape
        input = torch.cat((emb, price.unsqueeze(1)), dim = -1) #batch_len, emb_shape + 1
        
        input = self.fc(input) # batch_len, out_dim
        return input


class UnionMeanEmbeddingModel(CatMeanEmbeddingModel):
    def __init__(self):
        super(UnionMeanEmbeddingModel, self).__init__()

    # def set(self, *args):
    #     super(UnionMeanEmbeddingModel, self).set(*args)
    #     self.embedding = nn.Embedding(*self.embedding.weight.shape)
    #     torch.nn.init.xavier_uniform_(self.embedding.weight)
    #     return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len, price):
        emb = self._calc_emb(union_idxs, union_len) # batch_len, emb_shape
        input = torch.cat((emb, price.unsqueeze(1)), dim = -1) #batch_len, emb_shape + 1
        
        input = self.fc(input) # batch_len, out_dim
        return input


class MeanEmbeddingModel2(MeanEmbeddingModel):
    def __init__(self):
        super(MeanEmbeddingModel2, self).__init__()

    def set(self, word_embeddings, out_dim, hid_dim=256):
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                      freeze=True)
        self.fc = nn.Sequential(
            nn.Linear(self.embedding.weight.shape[-1] * 2 + 1, hid_dim),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hid_dim, out_dim)
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        return self


