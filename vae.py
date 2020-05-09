import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_shape):
        # Embedding size is hidden_shape[0]
        super().__init__()
        hidden_1, hidden_2 = hidden_shape
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_1)
        self.enc_mu = nn.Linear(hidden_1, hidden_2)
        self.enc_logvar = nn.Linear(hidden_1, hidden_2)
        
        self.fc_out1 = nn.Linear(hidden_2, hidden_1)
        self.fc_out2 = nn.Linear(hidden_1, hidden_1)
        self.out = nn.Linear(hidden_1, input_dim)
            
    def encode(self, x):
        hid = F.relu(self.fc1(x))
        hid = F.relu(self.fc2(hid))
        return self.enc_mu(hid), self.enc_logvar(hid)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        hid = F.relu(self.fc_out1(z))
        hid = F.relu(self.fc_out2(hid))
        return self.out(hid)
    
    def forward(self, t):
        mu, logvar = self.encode(t)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar