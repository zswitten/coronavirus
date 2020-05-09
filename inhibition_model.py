import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):

    def __init__(self, input_dim, hidden_shape):
        super(Net, self).__init__()
        hidden_shape1, hidden_shape2 = hidden_shape
        self.fc1 = nn.Linear(input_dim, hidden_shape1)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(hidden_shape1, hidden_shape2)
        self.out = nn.Linear(hidden_shape2, 1)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class NetWithEmbedding(nn.Module):

    def __init__(self, vae, hidden_shape):
        super(NetWithEmbedding, self).__init__()
        hidden_shape1, hidden_shape2 = hidden_shape
        self.vae = vae
        self.embedding = vae.fc1
        self.fc1 = nn.Linear(self.embedding.out_features, hidden_shape1)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(hidden_shape1, hidden_shape2)
        self.out = nn.Linear(hidden_shape2, 1)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.embedding(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x