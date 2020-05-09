import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition_model import Net
from make_training_data import make_data

train_x, train_y, valid_x, valid_y, test_x, test_y = make_data()

model = Net(input_dim=train_x.shape[1], hidden_shape=[64, 64])

def train_epoch(model, x, y):
    optimizer = model.optimizer
    for x0, y0 in zip(x, y):
        prediction = model(x0)
        loss = model.loss_func(prediction, y0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_model(model, train_x, train_y, valid_x, valid_y, epochs=100):
    print(
        'Training error',
        model.loss_func(model(valid_x), valid_y)
    )
    print(
        'Validation error',
        model.loss_func(model(train_x), train_y)
    )
    for epoch in range(epochs):
        train_epoch(model, train_x, train_y)
        print(
            'Training error',
            model.loss_func(model(train_x), train_y)
        )
        print(
            'Validation error',
            model.loss_func(model(valid_x), valid_y)
        )
    return model
