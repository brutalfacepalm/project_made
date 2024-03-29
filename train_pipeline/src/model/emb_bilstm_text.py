import numpy as np
import torch
from torch import nn


class UnionEmbeddingBiLSTM(nn.Module):
    def __init__(self):
        super(UnionEmbeddingBiLSTM, self).__init__()


    def set(self, word_embeddings, out_dim):
        self.n_classes = out_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                      freeze=True)
        self.lstm = nn.LSTM(input_size=self.embedding.weight.shape[-1], 
                            hidden_size=self.embedding.weight.shape[-1], 
                            batch_first=True,
                            bidirectional=True,
                            )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.embedding.weight.shape[-1], out_dim),
        )

        for m in self.fc:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        emb = self.embedding(union_idxs) # batch_len, seq_len, emb_shape

        out, (hidden, cell) = self.lstm(emb)

        output = hidden[1,:,:]
        output = self.fc(self.dropout(output))
        return output


class EmbeddingBiLSTM(nn.Module):
    def __init__(self):
        super(EmbeddingBiLSTM, self).__init__()

    def set(self, word_embeddings, out_dim, hid_dim=256, dropout=0.5):
        self.n_classes = out_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                        freeze=True)
        self.lstm = nn.LSTM(input_size=self.embedding.weight.shape[-1], 
                            hidden_size=self.embedding.weight.shape[-1], 
                            # num_layers = 2,
                            # dropout=0.5,
                            batch_first=True,
                            bidirectional=True,
                            )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.embedding.weight.shape[-1], hid_dim),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hid_dim, out_dim)
        )
        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        name_emb = self.embedding(name_idxs) # batch_len, seq_len, emb_shape
        desc_emb = self.embedding(desc_idxs) # batch_len, seq_len, emb_shape

        out_name, (hidden_name, cell_name) = self.lstm(name_emb)
        out_desc, (hidden_desc, cell_desc) = self.lstm(desc_emb)

        output = torch.cat((hidden_name[1,:,:],
                            hidden_desc[1,:,:],
                            ), dim=-1) #batch_len, 2*emb_shape
        output = self.fc(self.dropout(output))
        return output


class EmbeddingBiLSTM2(nn.Module):
    def __init__(self):
        super(EmbeddingBiLSTM2, self).__init__()

    def set(self, word_embeddings, out_dim, hid_dim=256):
        self.n_classes = out_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                        freeze=True)
        self.name_lstm = nn.LSTM(input_size=self.embedding.weight.shape[-1], 
                                hidden_size=self.embedding.weight.shape[-1], 
                                batch_first=True,
                                bidirectional=True,
                                )
        self.desc_lstm = nn.LSTM(input_size=self.embedding.weight.shape[-1], 
                                hidden_size=self.embedding.weight.shape[-1], 
                                batch_first=True,
                                bidirectional=True,
                                )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.embedding.weight.shape[-1], hid_dim),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hid_dim, out_dim)
        )
        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        name_emb = self.embedding(name_idxs) # batch_len, seq_len, emb_shape
        desc_emb = self.embedding(desc_idxs) # batch_len, seq_len, emb_shape

        out, (name_hidden, cell) = self.name_lstm(name_emb)
        out, (desc_hidden, cell) = self.desc_lstm(desc_emb)

        output = torch.cat((name_hidden[1,:,:],
                            desc_hidden[1,:,:],
                            ), dim=-1) #batch_len, 2*emb_shape

        output = self.fc(self.dropout(output))
        return output



