import torch


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            *inputs, output = batch
            inputs = [input.to(device) for input in inputs]
                
            pred = model(*inputs)
            loss = criterion(pred.cpu(), output)
            total_loss += loss.item()

    return total_loss / step


def train_epoch(model, dataloader, criterion, optimizer, device, n_step_print=1000):

    model.train()
    print('embedding requires_grad =', model.embedding.weight.requires_grad,
          'lr =', optimizer.param_groups[0]['lr'])
    sum_loss = 0.0
    for step, batch in enumerate(dataloader, start=1):
        *inputs, labels = batch
        inputs = [input.to(device) for input in inputs]
        
        optimizer.zero_grad()
        
        logits = model(*inputs).cpu()
    
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        sum_loss += float(loss.item())

        if step % n_step_print == 0:
            mean_loss = sum_loss / n_step_print
            sum_loss = 0.0
            print('     step {}/{}, loss: {:.3f}'.format(step, len(dataloader), mean_loss)) 

    return evaluate_epoch(model, dataloader, criterion, device)