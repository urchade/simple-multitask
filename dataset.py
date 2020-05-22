import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class RegressionData(Dataset):
    def __init__(self, n_samples, n_features, n_tasks):
        self.n_tasks = n_tasks
        self.n_samples = n_samples

        w = torch.randn(size=(n_features, n_tasks))
        b = torch.randn(size=(n_tasks,))

        self.x = torch.randn(size=(n_samples, n_features))
        self.y = self.x @ w + b

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        return x, y