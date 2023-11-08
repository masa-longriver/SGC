import torch
import torch.nn.functional as F

def run(config, model, dataset, optim, state='train'):
    if state == 'train':
        model.train()
        optim.zero_grad()
        pred = model(dataset.data)
        loss = F.nll_loss(
            pred[dataset.data.train_mask],
            dataset.data.y[dataset.data.train_mask]
        )
        loss.backward()
        optim.step()
    
    elif state == 'valid':
        model.eval()
        with torch.no_grad():
            pred = model(dataset.data)
            loss = F.nll_loss(
                pred[dataset.data.val_mask],
                dataset.data.y[dataset.data.val_mask]
            )
    
    elif state == 'test':
        model.eval()
        with torch.no_grad():
            pred = model(dataset.data)
            loss = F.nll_loss(
                pred[dataset.data.test_mask],
                dataset.data.y[dataset.data.test_mask]
            )
    
    return loss
