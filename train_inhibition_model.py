import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

from inhibition_model import Net
from make_training_data import make_data

# train_x, train_y, valid_x, valid_y, test_x, test_y = make_data()

# model = Net(input_dim=train_x.shape[1], hidden_shape=[64, 64])

class InhibitionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.to_list()

        x = self.df.loc[idx, 'gexp'].values
        x = torch.as_tensor(
            # np.array([row for row in x]),
            dtype=torch.float32
        )
        y = self.df.loc[idx, 'Inh_index']#.values
        y = torch.as_tensor(
            y, dtype=torch.float32
        )
        return {'gexp': x, 'inhibition': y}

# def make_data_loader(df, batch_size=32):
#     dataset = InhibitionDataset(df)
#     data_loader = DataLoader(dataset, batch_size=32)
#     return data_loader

# def train_epoch(model, x, y, optimizer):
#     for x0, y0 in zip(x, y):
#         prediction = model(x0)
#         loss = model.loss_func(prediction, y0)
#         optimizer.zero_grad()
#         loss.backward()
#         if random.random() < 0.0001:
#             print("Norms:", model.fc1.weight.grad.norm().item(), model.fc2.weight.grad.norm().item(), model.out.weight.grad.norm().item(), loss.item())
#         optimizer.step()

def train_epoch(model, x, y, optimizer, dataset, batch_size=32):
    pass

def train_model(model, train_x, train_y, valid_x, valid_y, epochs=100):
    print(
        'Training error',
        model.loss_func(model(valid_x), valid_y)
    )
    print(
        'Validation error',
        model.loss_func(model(train_x), train_y)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        train_epoch(model, train_x, train_y, optimizer)
        print(
            'Training error',
            model.loss_func(model(train_x), train_y).item()
        )
        print(
            'Validation error',
            model.loss_func(model(valid_x), valid_y).item()
        )
        print('\n')
    return model
