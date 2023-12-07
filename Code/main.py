import torch
from torch.utils.data import DataLoader

from dataset import FeedbackDataset


def main():
    tr_dataset = FeedbackDataset(cfg, train_fold)
    train_loader = DataLoader(tr_dataset, batch_size=cfg.dataset.batch_size,
                              num_workers=cfg.dataset.num_workers, shuffle=True,
                              pin_memory=True, drop_last=True)