import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim=42, dropout_rate=0.4):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(noise_dim, 128)
        self.fc2 = nn.Linear(128 + noise_dim, 256)
        self.fc3 = nn.Linear(256 + 128, 512)
        self.fc_out = nn.Linear(512, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(256)
        self.layer_norm3 = nn.LayerNorm(512)

    def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x), 0.1)
        x1 = self.layer_norm1(x1)
        x1 = self.dropout(x1)

        x2 = F.leaky_relu(self.fc2(torch.cat([x, x1], dim=1)), 0.1)
        x2 = self.layer_norm2(x2)
        x2 = self.dropout(x2)

        x3 = F.leaky_relu(self.fc3(torch.cat([x1, x2], dim=1)), 0.1)
        x3 = self.layer_norm3(x3)
        x3 = self.dropout(x3)

        return torch.tanh(self.fc_out(x3))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(42, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)
        self.layer_norm3 = nn.LayerNorm(128)

    def forward(self, x):
        x = F.leaky_relu(self.layer_norm1(self.fc1(x)), 0.1)
        x = F.dropout(x, 0.4)
        x = F.leaky_relu(self.layer_norm2(self.fc2(x)), 0.1)
        x = F.dropout(x, 0.4)
        x = F.leaky_relu(self.layer_norm3(self.fc3(x)), 0.1)
        x = F.dropout(x, 0.4)
        return torch.sigmoid(self.fc4(x))
