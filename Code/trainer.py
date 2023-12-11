import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm

from utils import AverageMeter, load_from_saved, mcrmse_labelwise_score
from dataset import collate_fn


def train_one_epoch(cfg, train_loader, model, criterion, optimizer, scheduler):

    losses = AverageMeter()
    scaler = GradScaler()
    model.train()

    train_loader = tqdm(train_loader, total=len(train_loader), desc='Training', leave=False)
    for step, batch in enumerate(train_loader):
        # print(batch)  # [{...}] that's why batch[0]
        inputs, labels = batch[0], batch[1]['labels']  # collate(batch[0])
        for k, v in inputs.items():
            inputs[k] = v.to(cfg.device)
        labels = labels.to(cfg.device)
        batch_size = labels.size(0)  # cannot use from cfg because of drop=True

        with autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        # scaler.unscale_(optimizer)  # Reverses the Scaling of Gradients, Allows for Gradient Clipping and Other Operations
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
        inputs, labels = batch[0], batch[1]['labels']  # collate(batch[0])
        for k, v in inputs.items():
            inputs[k] = v.to(cfg.device)
        labels = labels.to(cfg.device)
        batch_size = labels.size(0)

        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, labels)

        losses.update(loss.item(), batch_size)
        predictions.append(preds.detach().cpu())  # preds.to('cpu').numpy())
        labels_.append(labels.detach().cpu())  # labels.to('cpu').numpy())

    predictions = torch.cat(predictions).numpy()
    labels_ = torch.cat(labels_).numpy()

    average_mcrmse, individual_scores = mcrmse_labelwise_score(labels_, predictions)

    return losses.avg, average_mcrmse, individual_scores
