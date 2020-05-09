import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from vae import VAE

DATA_FILE = 'data/level5_1000.csv'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Lincs(Dataset):

    def __init__(self, df):
        super().__init__()
        self.df = df
    
    def shape(self):
        return self.df.shape
    
    def __len__(self):
        return self.df.shape[1]
    
    def __getitem__(self, idx):
        return torch.tensor(self.df.iloc[:,idx].values, dtype=torch.float32)

def loss_function(recon_x, x, mu, logvar):
    mu.to(DEVICE)
    mse = F.mse_loss(x, recon_x, reduction=True)
    kld = 0.5*(mu.pow(2).sum(dim=-1) + torch.exp(logvar).sum(dim=-1) - (logvar+1).sum(dim=-1))
    
    return (mse + kld).sum(dim=-1)

def train_epoch(model, data_loader, optimizer):
    
    model.train()
    train_loss = 0
    for batch_idx, x in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.to(DEVICE)
        recon_batch, mu, logvar = model(x)
        loss = loss_function(recon_batch, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(data_loader)

def run_training(model, data_loader, optimizer, epochs=1000):

    train_losses = []
    for epoch in range(epochs):
        train_losses.append(train_epoch(model, data_loader, optimizer))
        if epoch % 2 == 0:
            print(f'=======> Epoch: {epoch} Average loss: {train_losses[-1]}')
    #plt.plot(np.arange(len(train_losses)), train_losses)

def load_data(location):
    df = pd.read_csv(location)
    df = df.set_index('rid')
    return df

def make_data_loader(df, batch_size=32):
    lincs = Lincs(df)
    data_loader = DataLoader(lincs, batch_size=32)
    return data_loader

def make_adam_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-3)

def train_vae(epochs=1000, batch_size=32, model=None):
    dims = [64, 7]

    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    data_df = load_data(DATA_FILE)

    data_loader = make_data_loader(data_df)
    if not model:
        model = VAE(
            input_dim=len(data_df),
            hidden_shape=dims
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    optimizer = make_adam_optimizer(model)

    run_training(model, data_loader, optimizer, epochs=epochs)
    return model

