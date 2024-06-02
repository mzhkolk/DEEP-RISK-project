#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.0, stride=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, padding=2, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample_block = None

        if self.conv1.in_channels != self.conv2.out_channels:
            self.downsample_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels))

        # Apply He initialization to convolutional layers
        for layer in [self.conv1, self.conv2]:
            xavier_normal_(layer.weight)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        if out.shape != residual.shape:
            out = F.interpolate(out, size=(residual.shape[-2], residual.shape[-1]), mode='bilinear', align_corners=False)
        if self.downsample_block is not None:
            residual = self.downsample_block(residual)
        out += residual
        out = self.relu(out)
        return out.to(x.device)

class VAE(nn.Module):
    def __init__(self, latent_dim, red):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.red = 4

        # Encoder layers
        encoder_layers = []
        encoder_layers.append(nn.Conv2d(1, 8 * red, kernel_size=4, stride=1, padding=1, dilation=2))
        encoder_layers.append(nn.BatchNorm2d(8 * red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(8 * red, 8 * red, kernel_size=4, stride=1))

        encoder_layers.append(nn.Conv2d(8 * red, 16 * red, kernel_size=4, stride=1, padding=1, dilation=2))
        encoder_layers.append(nn.BatchNorm2d(16 * red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(16 * red, 16 * red, kernel_size=4, stride=1))

        encoder_layers.append(nn.Conv2d(16 * red, 32 * red, kernel_size=4, stride=1, padding=1, dilation=1))
        encoder_layers.append(nn.BatchNorm2d(32 * red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(32 * red, 32 * red, kernel_size=4, stride=1))
   

        encoder_layers.append(nn.Conv2d(32 * red, 64 * red, kernel_size=4, stride=1, padding=1, dilation=1))
        encoder_layers.append(nn.BatchNorm2d(64 * red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(64 * red, 64 * red, kernel_size=4, stride=1))
        
        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 12 ,210)
            encoder_output_shape, decoder_input_shape = self._get_encoder_output_shape(dummy_input)
        self.fc_mu = nn.Linear(encoder_output_shape, self.latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_shape, self.latent_dim)

        # Decoder layers
        decoder_layers = []
        self.decoder_input = nn.Linear(self.latent_dim, encoder_output_shape)
        self.decoder_reshape = decoder_input_shape

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        decoder_layers.append(nn.Conv2d(64 * red, 32 * red, kernel_size=4, stride=1, padding=1))
        decoder_layers.append(nn.BatchNorm2d(32 * red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(32 * red, 32 * red, kernel_size=4, stride=1))
        
        
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        decoder_layers.append(nn.Conv2d(32 * red, 16 * red, kernel_size=4, stride=1, padding=1))
        decoder_layers.append(nn.BatchNorm2d(16 * red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(16 * red, 16 * red, kernel_size=4, stride=1))

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        decoder_layers.append(nn.Conv2d(16 * red, 8 * red, kernel_size=4, stride=1, padding=1))
        decoder_layers.append(nn.BatchNorm2d(8 * red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(8 * red, 8 * red, kernel_size=4, stride=1))

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))#        decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(nn.Conv2d(8 * red, 1, kernel_size=4, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)
        for layer in encoder_layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        batch_size = x.size(0)
        decoder_reshape_without_batch = self.decoder_reshape[1:]
        x = x.view(batch_size, *decoder_reshape_without_batch)
        x_recon = self.decoder(x)
        x_recon = F.interpolate(x_recon, size=(12, 210), mode='bilinear', align_corners=False)
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var

    def _get_encoder_output_shape(self, x):
        x = self.encoder(x)
        encoder_output_shape = x.view(x.size(0), -1).shape[1]
        decoder_input_shape = x.view(x.size(0), x.size(1), x.size(2), x.size(3)).shape
        return encoder_output_shape, decoder_input_shape

class VAE_Loss(nn.Module):
    def __init__(self, beta):
        super(VAE_Loss, self).__init__()
        self.beta = beta

    def forward(self, decoded, x, mu, log_var):
        reconstruction_loss = F.mse_loss(decoded, x, reduction='sum')
        kl_divergence_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss
        return total_loss, reconstruction_loss, kl_divergence_loss

