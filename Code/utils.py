import random
import os
import math
import numpy as np
import torch
from sklearn.metrics import mean_squared_error


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_differential_lr_parameters(model, encoder_lr, decoder_lr, embeddings_lr, lr_mult_factor, weight_decay, num_groups):
    # def create_layer_groups(num_layers, num_groups):
    num_layers = model.backbone_config.num_hidden_layers
    step = math.ceil(num_layers / num_groups)
    layer_groups = []
    for start in range(0, num_layers, step):
        end = min(start + step, num_layers)
        group = [f'backbone.encoder.layer.{num_layers - i - 1}.' for i in range(start, end)]
        layer_groups.append(group)
        # return layer_groups

    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
    # layer_groups = create_layer_groups(n_layers, n_groups)

    for name, params in named_parameters:
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        lr = None

        if name.startswith("backbone.encoder"):
            lr = next((encoder_lr * (lr_mult_factor ** (i + 1)) for i, group in enumerate(layer_groups) if any(layer in name for layer in group)), encoder_lr)
        elif name.startswith("backbone.embeddings"):
            lr = embeddings_lr
        elif name.startswith("fc") or name.startswith('backbone.pooler'):
            lr = decoder_lr

        if lr is not None:
            opt_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

    return opt_parameters


# https://www.kaggle.com/code/nischaydnk/fb3-pytorch-lightning-training-baseline
def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "backbone" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def load_from_saved(model, checkpoint_path, optimizer=None, scheduler=None):
    state = torch.load(checkpoint_path, map_location='cuda')  # 'cpu'
    if 'model.embeddings.position_ids' in state['model'].keys():
        new_state = {}
        for key, value in state['model'].items():
            new_key = key
            if key.startswith('model.'):
                # because we are calling it backbone in our CustomModel class.
                new_key = key.replace('model', 'backbone')
            new_state[new_key] = value

        state['model'] = new_state

    model.load_state_dict(state['model'])
    if optimizer and scheduler is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

        return model, optimizer, scheduler, state['epoch']

    return model, state['epoch']


def mcrmse_labelwise_score(true_values, predicted_values):
    individual_scores = []
    num_targets = true_values.shape[1]  # Assuming true_values is 2D: (num_samples, num_targets)

    for target_index in range(num_targets):
        true_target = true_values[:, target_index]
        predicted_target = predicted_values[:, target_index]
        mse_score = mean_squared_error(true_target, predicted_target, squared=False)
        individual_scores.append(mse_score)

    average_mcrmse = np.mean(individual_scores)
    return average_mcrmse, individual_scores


# https://github.com/sanchitvj/rsppUnet-BraTS-2021/blob/77ab8524e9684a31f835ca9972c485120496a34d/src/utils/ops.py#L12
def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


def delete_model_file(file_path):
    # Check if file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"The file '{file_path}' does not exist.")


def round_to_half(number):
    return round(number * 2) / 2
