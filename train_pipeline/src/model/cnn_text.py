import numpy as np
import torch
from torch import nn


class CnnInterface(nn.Module):
    def __init__(self):
        super(CnnInterface, self).__init__()

    def set(self, word_embeddings, out_dim):
        self.n_classes = out_dim
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(word_embeddings)), 
                                                      freeze=True)

class TextCnnModel(CnnInterface):
    def __init__(self):
        super().__init__()

    def set(self, word_embeddings, out_dim):
        super().set(word_embeddings, out_dim)

        emb_len = self.embedding.weight.shape[-1]
        
        self.cnn1 = nn.Conv1d(emb_len, 200, 1)
        self.cnn2 = nn.Conv1d(emb_len, 200, 2)
        self.cnn3 = nn.Conv1d(emb_len, 200, 3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(600, out_dim)

        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        embedded = self.embedding(union_idxs) # batch_len, seq_len, emb_shape
        embedded = embedded.permute(0, 2, 1) # batch_len, emb_shape, seq_len
        output1 = self.pool(self.cnn1(embedded))
        output2 = self.pool(self.cnn2(embedded))
        output3 = self.pool(self.cnn3(embedded)) # batch_len, 200, 1
        output = torch.cat([output1, output2, output3], dim=1).squeeze(-1) #batch_len, emb_shape*2
        output = self.fc(self.drop(output))
        return output
        

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reorder(nn.Module):
    def forward(self, x):
        return x.permute((0, 2, 1))


class SimpleCnnModel(CnnInterface):
    def __init__(self):
        super(SimpleCnnModel, self).__init__()

    def set(self, word_embeddings, out_dim):
        super(SimpleCnnModel, self).set(word_embeddings, out_dim)

        n_maximums = 2
        hid_size = self.embedding.weight.shape[-1]

        simple_model = nn.Sequential()
        simple_model.add_module('emb', self.embedding)
        simple_model.add_module('reorder', Reorder())
        simple_model.add_module('conv1', nn.Conv1d(
                                    in_channels=hid_size,
                                    out_channels=hid_size*2,
                                    kernel_size=2
                                ))
        simple_model.add_module('relu1', nn.ReLU())
        simple_model.add_module('conv2', nn.Conv1d(
                                    in_channels=hid_size*2,
                                    out_channels=hid_size*2,
                                    kernel_size=3
                                ))
        simple_model.add_module('bn1', nn.BatchNorm1d(hid_size*2))
        simple_model.add_module('relu2', nn.ReLU())
        simple_model.add_module('adaptive_pool', nn.AdaptiveMaxPool1d(n_maximums))
        simple_model.add_module('flatten', nn.Flatten())
        simple_model.add_module('dropout', nn.Dropout(0.5))
        simple_model.add_module('out', nn.Linear(hid_size*2*n_maximums, out_dim))  

        self.model = simple_model

        for m in self.model:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        
        return self

    def forward(self, name_idxs, name_len, desc_idxs, desc_len, union_idxs, union_len):
        logits = self.model(union_idxs)
        return logits






