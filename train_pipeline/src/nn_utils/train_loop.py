import torch
import numpy as np



def evaluate(model, dataloader, criterion, device):
    model.eval()

    n_classes = model.n_classes
    target_true = np.zeros(n_classes)
    predict_true = np.zeros(n_classes)
    correct_true = np.zeros(n_classes)

    with torch.no_grad():
        total_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
                
            logits = model(*inputs).cpu()
            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            y_pred = torch.argmax(logits, dim = -1).numpy()
            y_true = labels.numpy()
            for i in range(n_classes):
                target_true[i] += np.sum((y_true == i).astype(int))
                predict_true[i] += np.sum((y_pred == i).astype(int))
                correct_true[i] += np.sum((y_true == i).astype(int) * (y_pred == i).astype(int))

    recall_classes = np.zeros(n_classes)
    precision_classes = np.zeros(n_classes)
    for i in range(n_classes):
        recall_classes[i] = correct_true[i] / target_true[i] if target_true[i] != 0 else 0
        precision_classes[i] = correct_true[i] / predict_true[i] if predict_true[i] != 0 else 0
    macro_avg_precision = np.mean(precision_classes)
    macro_avg_recall = np.mean(recall_classes)

    return total_loss / step, macro_avg_precision, macro_avg_recall


def train_epoch(model, dataloader, criterion, optimizer, device, n_step_print=200):

    model.train()
    print('embedding requires_grad =', model.embedding.weight.requires_grad,
          'lr =', optimizer.param_groups[0]['lr'])
    sum_loss = 0.0
    for step, batch in enumerate(dataloader, start=1):
        *inputs, labels = batch
        inputs = [input.to(device) for input in inputs]
        
        optimizer.zero_grad()
        
        logits = model(*inputs).cpu()
        # import pdb
        # pdb.set_trace()
        # max_rel = ((torch.nn.functional.one_hot(labels, model.n_classes) * logits).sum(dim=1) \
        #             / logits.max(1).values 
        #             ).sum()
        loss = criterion(logits, labels)# - 0.01 * max_rel

              

        loss.backward()
        optimizer.step()

        sum_loss += float(loss.item())

        if step % n_step_print == 0:
            mean_loss = sum_loss / n_step_print
            sum_loss = 0.0
            print('     step {}/{}, loss: {:.3f}'.format(step, len(dataloader), mean_loss)) 

