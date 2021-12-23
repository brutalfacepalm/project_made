import numpy as np
import torch
from torch import nn


class MeanEmbeddingInterface(nn.Module):
    def __init__(self):
        super(MeanEmbeddingInterface, self).__init__()

    def set(self, word_embeddings, out_dim):
        self.n_classes = out_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                      freeze=True)

    def _calc_emb(self, text_idxs, text_len):
        emb = self.embedding(text_idxs) # batch_len, seq_len, emb_shape
        emb = torch.sum(emb, dim=1) # batch_len, emb_shape
        emb = nn.functional.normalize(emb, dim=1)
        # text_len[text_len == 0] = 1
        # emb = torch.div(emb, text_len.unsqueeze(1)) # batch_len, emb_shape
        return emb


class UnionMeanEmbeddingModel(MeanEmbeddingInterface):
    def __init__(self):
        super(UnionMeanEmbeddingModel, self).__init__()

    def set(self, word_embeddings, out_dim):
        super(UnionMeanEmbeddingModel, self).set(word_embeddings, out_dim)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.embedding.weight.shape[-1], out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        emb = self._calc_emb(union_idxs, union_len) # batch_len, emb_shape
        logits = self.fc(self.dropout(emb)) # batch_len, out_dim
        return logits


class CatMeanEmbeddingModel(MeanEmbeddingInterface):
    def __init__(self):
        super(CatMeanEmbeddingModel, self).__init__()

    def set(self, word_embeddings, out_dim):
        super(CatMeanEmbeddingModel, self).set(word_embeddings, out_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.embedding.weight.shape[-1] * 2, out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        name_emb = self._calc_emb(name_idxs, name_len) # batch_len, emb_shape
        desc_emb = self._calc_emb(desc_idxs, desc_len) # batch_len, emb_shape
        cat_emb = torch.cat((name_emb, desc_emb), dim = -1) #batch_len, 2 * emb_shape
        
        logits = self.fc(self.dropout(cat_emb)) # batch_len, out_dim
        return logits


class CatMeanEmbeddingModel2(CatMeanEmbeddingModel):
    def __init__(self):
        super(CatMeanEmbeddingModel2, self).__init__()

    def set(self, word_embeddings, out_dim, hid_dim=256):
        super(CatMeanEmbeddingModel2, self).set(word_embeddings, out_dim)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.embedding.weight.shape[-1] * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        return self


