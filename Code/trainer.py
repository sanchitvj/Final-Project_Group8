import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm

from utils import AverageMeter, load_from_saved
from dataset import collate_fn


def train_one_epoch(cfg, train_loader, model, criterion, optimizer, scheduler):

    losses = AverageMeter()
    scaler = GradScaler()
    model.train()

    for step, batch in enumerate(train_loader):

        inputs, labels = collate_fn(batch)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)  # cannot use from cfg because of drop=True

        with autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)  # Reverses the Scaling of Gradients, Allows for Gradient Clipping and Other Operations
        clip_grad_norm(model.parameters(), cfg.training.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return losses.avg


def validation(cfg, val_loader, model, criterion):

    losses = AverageMeter()
    model.eval()
    predictions = []
    labels_ = []

    for step, batch in enumerate(val_loader):
        inputs, labels = collate_fn(batch)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, labels)

        losses.update(loss.item(), batch_size)
        predictions.append(y_preds.to('cpu').numpy())
        labels_.append(labels.to('cpu').numpy())

    predictions = np.concatenate(predictions)
    labels_ = np.concatenate


    return losses.avg, predictions








