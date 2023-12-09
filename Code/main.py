import os
import warnings
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from termcolor import colored

from transformers import AutoConfig, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from dataset import FeedbackDataset, collate_fn
from model import CustomModel
from utils import load_from_saved, mcrmse_labelwise_score, seed_torch, Config, delete_model_file
from trainer import train_one_epoch, validation

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(cfg):

    print(colored("GPU Name: {}".format(torch.cuda.get_device_name(0)), "green"))

    seed_torch(cfg.seed)
    df = pd.read_csv(cfg.dataset.data_path)
    train, valid = train_test_split(df, test_size=0.2, random_state=cfg.seed)

    wandb.login(key=cfg.logger.key)
    wandb.init(project=cfg.logger.project_name)
    wandb.run.name = cfg.model.backbone_name.split('/')[1] + '_' + cfg.model.pooling + '_' + str(cfg.training.epochs)
    if cfg.logger.save:
        wandb.run.save()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    cfg.tokenizer = tokenizer

    tr_dataset = FeedbackDataset(cfg, train)
    vl_dataset = FeedbackDataset(cfg, valid)

    train_loader = DataLoader(tr_dataset, batch_size=cfg.dataset.tr_batch_size,
                              num_workers=cfg.dataset.num_workers, shuffle=True,
                              pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(vl_dataset, batch_size=cfg.dataset.vl_batch_size,
                            num_workers=cfg.dataset.num_workers, shuffle=False,
                            pin_memory=True, drop_last=False, collate_fn=collate_fn)

    backbone_config = AutoConfig.from_pretrained(cfg.model.backbone_name, output_hidden_states=True)
    model = CustomModel(cfg, backbone_config)
    print(
        colored(
            f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            "green",
        )
    )

    optimizer = Adam(model.parameters(), lr=float(cfg.training.encoder_lr),
                     betas=tuple(float(beta) for beta in cfg.training.betas),
                     eps=float(cfg.training.eps), weight_decay=float(cfg.training.weight_decay))

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_cycles=0.5, num_warmup_steps=0,
                                                num_training_steps=(
                                                            cfg.training.epochs + round(cfg.training.epochs * 0.5)))
    last_epoch = 0
    if cfg.resume_train:
        model, optimizer, scheduler, last_epoch = load_from_saved(model, optimizer, scheduler,
                                                                  cfg.model.saved_model_path)
        print(f">>>> Checkpoint loaded from epoch {last_epoch} <<<<")

    model.to(cfg.device)
    criterion = nn.MSELoss()
    best_score = np.inf
    for epoch in range(last_epoch + 1, cfg.training.epochs):

        tr_loss = train_one_epoch(cfg, train_loader, model, criterion, optimizer, scheduler)
        vl_loss, avg_mcrmse, individual_scores = validation(cfg, val_loader, model, criterion)

        tqdm.write(
            f"Epoch {epoch}/{cfg.training.epochs} - Train Loss: {tr_loss:.3f}, Val Loss: {vl_loss:.4f}, Avg MCRMSE: {avg_mcrmse:.4f}")

        prev_saved = []
        if avg_mcrmse < best_score:
            best_score = avg_mcrmse
            print(f"Checkpoint saved at best score: {best_score:.4f}")

            if not os.path.exists('saved_checkpoints'):
                os.makedirs('saved_checkpoints')

            filepath = os.path.join('saved_checkpoints',
                                    f"{cfg.model.backbone_name.split('/')[1]}_{cfg.model.pooling}_{epoch}_{best_score:.4f}.pth")
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.step(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}, filepath)
            if len(prev_saved) > 0:
                delete_model_file(prev_saved[0])
                prev_saved.pop(0)
            prev_saved.append(filepath)

        wandb.log({f"Epoch": epoch,
                   f"avg_train_loss": tr_loss,
                   f"avg_val_loss": vl_loss,
                   f"Avg MCRMSE Loss": avg_mcrmse,
                   f"Cohesion rmse": individual_scores[0],
                   f"Syntax rmse": individual_scores[1],
                   f"Vocabulary rmse": individual_scores[2],
                   f"Phraseology rmse": individual_scores[3],
                   f"Grammar rmse": individual_scores[4],
                   f"Conventions rmse": individual_scores[5]})

    wandb.finish()


if __name__ == "__main__":

    try:
        with open("config.yaml", 'r') as yaml_file:
            cfg_dict = yaml.safe_load(yaml_file)
            cfg = Config(cfg_dict)
    except FileNotFoundError:
        print("The YAML configuration file was not found.")
    except yaml.YAMLError as exc:
        print("Error parsing YAML file:", exc)
    except Exception as exc:
        print("An unexpected error occurred:", exc)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = device
    main(cfg)
