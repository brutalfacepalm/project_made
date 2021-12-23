import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

import torch
from torch import nn
# import gc

from nn_utils.dataloader_gen import dataloader_gen, weighted_dataloader_gen
from transform.vocabulary import Vocabulary
from nn_utils.train_loop import train_epoch, evaluate
   

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)       


class EmbeddingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 emb_model=None,#=UnionMeanEmbeddingModel(),
                 embedding_fun=None,#=CompressFastTextEmb(),
                 lr=1e-3,
                 weight_decay=0,
                 n_epoch=20,
                 n_unfreeze=0,
                 scheduler_step=10,
                 scheduler_gamma=0.5,
                 weighted_sampler=True,
                 weight_adding=0,
                 batch_size=1024,
                 ):
        self.model = emb_model
        self.embedding_fun = embedding_fun
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epoch = n_epoch
        self.n_unfreeze =  n_unfreeze
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.weighted_sampler = weighted_sampler
        self.weight_adding = weight_adding
        self.batch_size = batch_size


    def fit(self, X, y, X_val=None, y_val=None, y_enc=None):
        self.classes_ = y_enc.classes_ if y_enc else unique_labels(y)
        
        vocab = Vocabulary(X['product_description'], X['name_dish'])

        word_embeddings = self.embedding_fun(vocab.words)

        pad_emb = np.zeros(word_embeddings[0].shape)
        pad_idx = len(word_embeddings)
        word_embeddings.append(pad_emb)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        model = self.model.set(word_embeddings, len(self.classes_)).to(device)


        class_weights = len(y) / len(np.unique(y)) / np.bincount(y)
        class_weights += self.weight_adding

        if self.weighted_sampler:
            train_criterion = nn.CrossEntropyLoss()
            train_dataloader = weighted_dataloader_gen(X, y, vocab.text_to_idxs, pad_idx, 
                                                    weights=class_weights[y], batch_size=self.batch_size)
        else:
            train_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights), reduction='mean')
            train_dataloader = dataloader_gen(X, y, vocab.text_to_idxs, pad_idx, batch_size=self.batch_size, shuffle=True)
        
        valid_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights), reduction='mean')
        valid_dataloader = dataloader_gen(X_val, y_val, vocab.text_to_idxs, pad_idx, batch_size=2*self.batch_size, shuffle=False) \
                               if X_val is not None and y_val is not None else \
                               dataloader_gen(X, y, vocab.text_to_idxs, pad_idx, batch_size=2*self.batch_size, shuffle=False)


        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, 
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.scheduler_step, 
                                                    gamma=self.scheduler_gamma)

        # gc.collect()
        # torch.cuda.empty_cache()
        for epoch in range(self.n_epoch):
            logger.debug(f"Epoch {epoch}/{self.n_epoch}:")
            train_epoch(model, train_dataloader, train_criterion, optimizer, device)

            loss, precission, recall = \
                evaluate(model, valid_dataloader, valid_criterion, device)
            logger.debug(
                f"Evaluate validation: \n                              "
                f"loss={loss:.3f}, pres={precission:.3f}, recall={recall:.3f}"
            )
            if self.n_unfreeze > 0 and (epoch + 1) >= self.n_epoch - self.n_unfreeze:
                #unfreeze embeddings
                model.embedding.weight.requires_grad = True
            scheduler.step()

 
        self.model = model
        self.device = device
        self.vocab = vocab
        self.pad_idx = pad_idx


    def predict_proba(self, X):
        dataloader = dataloader_gen(X, None, self.vocab.text_to_idxs, self.pad_idx, batch_size=2048, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_prob = []
            for batch in dataloader:
                inputs = [input.to(self.device) for input in batch]
                logits = self.model(*inputs).cpu()
                prob = torch.nn.functional.softmax(logits, dim=-1)
                y_prob.extend(prob.tolist())
        return np.array(y_prob)


    def predict(self, X):
        y_prob = self.predict_proba(X)
        return np.argmax(y_prob, axis=-1)
        # dataloader = dataloader_gen(X, None, self.vocab.text_to_idxs, self.pad_idx, batch_size=2048, shuffle=False)
        # self.model.eval()
        # with torch.no_grad():
        #     y_pred = []
        #     for batch in dataloader:
        #         inputs = [input.to(self.device) for input in batch]
        #         pred = self.model(*inputs)
        #         y_pred.extend(torch.argmax(pred.cpu(), dim = -1).tolist())

        # return np.array(y_pred)

    


