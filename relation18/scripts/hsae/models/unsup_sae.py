"""Plain ReLU SAE used by the two unsupervised baselines (10k features, recon + L1)."""
import torch.nn as nn
import torch.nn.functional as F


class TraditionalSAE(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        return z, self.decoder(z)
