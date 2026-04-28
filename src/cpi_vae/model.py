"""ConvVAE model definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, z_dim=128, base_channels=64):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1),
            nn.ReLU(True),
        )
        self._feat_dim = c * 8 * 4 * 4
        self.fc_mu = nn.Linear(self._feat_dim, z_dim)
        self.fc_logvar = nn.Linear(self._feat_dim, z_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, z_dim=128, base_channels=64):
        super().__init__()
        c = base_channels
        self.fc = nn.Linear(z_dim, c * 8 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(c, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        return self.deconv(h)


class ConvVAE(nn.Module):
    def __init__(self, in_channels=3, z_dim=128, base_channels=64):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, z_dim, base_channels)
        self.decoder = ConvDecoder(in_channels, z_dim, base_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
