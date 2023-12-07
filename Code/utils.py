import torch


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


def make_folds(df, target_cols, n_splits, random_state):
    kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for n, (train_index, val_index) in enumerate(kfold.split(df, df[target_cols])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df


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
    return model, optimizer, scheduler
