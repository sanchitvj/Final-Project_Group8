import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW

from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

from dataset import FeedbackDataset
from model import CustomModel
from utils import load_from_saved, mcrmse_labelwise_score
from trainer import train_one_epoch, validation

def main(cfg):
    tr_dataset = FeedbackDataset(cfg, train_fold)
    vl_dataset = FeedbackDataset(cfg, valid_fold)

    train_loader = DataLoader(tr_dataset, batch_size=cfg.dataset.batch_size,
                              num_workers=cfg.dataset.num_workers, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(vl_dataset, batch_size=cfg.dataset.batch_size,
                            num_workers=cfg.dataset.num_workers, shuffle=False,
                            pin_memory=True, drop_last=False)

    backbone_config = AutoConfig.from_pretrained(cfg.model.backbone_name, output_hidden_states=True)
    model = CustomModel(cfg, backbone_config)

    optimizer = Adam(model.parameters(), lr=cfg.training.encoder_lr, betas=cfg.training.betas,
                     eps=cfg.training.eps, weight_decay=cfg.training.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_cycles=0.5,
                                                num_training_steps=(cfg.training.epochs + round(cfg.training.epochs * 0.5)))
    last_epoch = 0
    if cfg.saved_model_path is not None:
        model, optimizer, scheduler, last_epoch = load_from_saved(model, optimizer, scheduler, cfg.saved_model_path)
        print(f">>>> Checkpoint loaded from epoch {last_epoch} <<<<")

    criterion = nn.MSELoss()

    best_score = np.inf
    for epoch in range(last_epoch+1, cfg.training.epochs):

        tr_loss = train_one_epoch(cfg, train_loader, model, criterion, optimizer, scheduler)
        vl_loss, predictions = validation(cfg, val_loader, model, criterion)

        average_mcrmse, individual_scores = mcrmse_labelwise_score()










