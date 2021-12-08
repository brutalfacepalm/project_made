import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

import torch
from torch import nn
# import gc

from nn_utils.word_embedding import CompressFastTextEmb
from nn_utils.dataloader_gen import dataloader_gen, weighted_dataloader_gen
from transform.vocabulary import Vocabulary
from nn_utils.train_loop import train_epoch, evaluate_epoch
from model.mean_emb_nn import MeanEmbeddingModel
     

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)       


class EmbeddingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 emb_model=MeanEmbeddingModel(),
                 embedding_fun=CompressFastTextEmb(),
                 lr=1e-3,
                 n_epoch=20,
                 n_unfreeze=0,
                 scheduler_step=10,
                 scheduler_gamma=0.5,
                 weighted_sampler=True
                 ):
        self.model = emb_model
        self.embedding_fun = embedding_fun
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_unfreeze =  n_unfreeze
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.weighted_sampler = weighted_sampler


    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_ = unique_labels(y)

        vocab = Vocabulary(X['product_description'], X['name_dish'])

        word_embeddings = self.embedding_fun(vocab.words)

        pad_emb = np.zeros(word_embeddings[0].shape)
        pad_idx = len(word_embeddings)
        word_embeddings.append(pad_emb)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        model = self.model.set(word_embeddings, len(self.classes_)).to(device)


        class_weights = len(y) / len(np.unique(y)) / np.bincount(y)

        if self.weighted_sampler:
            criterion = nn.CrossEntropyLoss()
            train_dataloader = weighted_dataloader_gen(X, y, vocab.text_to_idxs, pad_idx, 
                                                    weights=class_weights[y], batch_size=64)
            valid_dataloader = weighted_dataloader_gen(X_val, y_val, vocab.text_to_idxs, pad_idx, 
                                                    weights=class_weights[y_val], batch_size=1024)
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights), reduction='mean')
            train_dataloader = dataloader_gen(X, y, vocab.text_to_idxs, pad_idx, batch_size=64, shuffle=True)
            valid_dataloader = dataloader_gen(X_val, y_val, vocab.text_to_idxs, pad_idx, batch_size=1024, shuffle=False) \
                            if X_val is not None and y_val is not None else None


        optimizer = torch.optim.Adam(model.fc.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.scheduler_step, 
                                                    gamma=self.scheduler_gamma)

        # gc.collect()
        # torch.cuda.empty_cache()
        for epoch in range(self.n_epoch):
            logger.debug(f"Epoch {epoch}")
            train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
            logger.debug(f"Train loss: {train_loss}")
            if valid_dataloader:
                eval_loss = evaluate_epoch(model, valid_dataloader, criterion, device)
                logger.debug(f"Eval loss: {eval_loss}")
            if self.n_unfreeze > 0 and (epoch + 1) >= self.n_epoch - self.n_unfreeze:
                #unfreeze embeddings
                model.embedding.weight.requires_grad = True
            scheduler.step()

 
        self.model = model
        self.device = device
        self.vocab = vocab
        self.pad_idx = pad_idx


    def predict(self, X):

        dataloader = dataloader_gen(X, None, self.vocab.text_to_idxs, self.pad_idx, batch_size=1024, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            for batch in dataloader:
                inputs = [input.to(self.device) for input in batch]
                pred = self.model(*inputs)
                y_pred.extend(torch.argmax(pred.cpu(), dim = -1).tolist())

        return np.array(y_pred)