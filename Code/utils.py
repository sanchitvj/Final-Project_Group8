import random
import os
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


# def make_folds(df, target_cols, n_splits, random_state):
#     kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     for n, (train_index, val_index) in enumerate(kfold.split(df, df[target_cols])):
#         df.loc[val_index, 'fold'] = int(n)
#     df['fold'] = df['fold'].astype(int)
#     return df


def load_from_saved(model, optimizer, scheduler, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cuda')  # 'cpu'
    if 'model.embeddings.position_ids' in state['model'].keys():
        new_state = {}
        for key, value in state['model'].items():
            new_key = key
            if key.startswith('model.'):
                # because we are calling it backbone in our CustomModel class.
                new_key = key.replace('model', 'backbone')
            new_state[new_key] = value

        updated = {'model': new_state}

    model.load_state_dict(updated['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    return model, optimizer, scheduler, state['epoch']


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
