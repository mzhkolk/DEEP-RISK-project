import pickle
import torch.nn.utils as torch_utils
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init
import pandas as pd
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.0, stride=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride)
        xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample_block = None
        
        if self.conv1.in_channels != self.conv2.out_channels:
            self.downsample_block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels))

        # Apply He initialization to convolutional layers
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)    
        if out.shape != residual.shape:
            residual = F.interpolate(residual, size=out.shape[-1], mode='nearest')      
        if self.downsample_block is not None:
            residual = self.downsample_block(residual)
        out += residual
        out = self.relu(out)
        return out.to(x.device)


class VAE(nn.Module):
    def __init__(self, latent_dim, red, drop):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.red = red
        self.drop = drop
        # Encoder layers
        encoder_layers = []
        encoder_layers.append(nn.Conv3d(12, 64 * red, kernel_size=4, stride=2, padding=2))
        encoder_layers.append(nn.BatchNorm3d(64* red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(64* red, 64* red, kernel_size=3, stride=1, dropout_rate=drop))
        
        encoder_layers.append(nn.Conv3d(64* red, 128* red, kernel_size=4, stride=2, padding=2))
        encoder_layers.append(nn.BatchNorm3d(128* red))
        encoder_layers.append(nn.ReLU(inplace=False)) 
        encoder_layers.append(ResidualBlock(128* red, 128* red, kernel_size=3, stride=1, dropout_rate=drop))
        
        encoder_layers.append(nn.Conv3d(128* red, 256* red, kernel_size=4, stride=2, padding=2))
        encoder_layers.append(nn.BatchNorm3d(256* red))
        encoder_layers.append(nn.ReLU(inplace=True))
        encoder_layers.append(ResidualBlock(256* red, 256* red, kernel_size=3, stride=1, dropout_rate=drop))
        encoder_layers.append(nn.ReLU(inplace=False))

        encoder_layers.append(nn.Conv3d(256* red, 512* red, kernel_size=4, stride=2, padding=2))
        encoder_layers.append(nn.BatchNorm3d(512* red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(512* red, 512* red, kernel_size=3, stride=1, dropout_rate=drop))
            
        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, 64, 64, 2)
            encoder_output_shape, decoder_input_shape = self._get_encoder_output_shape(dummy_input)

        self.fc_mu = nn.Linear(encoder_output_shape, self.latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_shape, self.latent_dim)
        
        # Decoder layers
        decoder_layers = []
        self.decoder_input = nn.Linear(self.latent_dim, encoder_output_shape)
        self.decoder_reshape = decoder_input_shape
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(nn.Conv3d(512* red, 256* red, kernel_size=4, stride=1, padding=1)) 
        decoder_layers.append(nn.BatchNorm3d(256* red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(256* red, 256* red, kernel_size=3, stride=1, dropout_rate=drop))

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(nn.Conv3d(256* red, 128* red, kernel_size=4, stride=1, padding=2)) 
        decoder_layers.append(nn.BatchNorm3d(128* red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(128* red, 128* red, kernel_size=3, stride=1, dropout_rate=drop))
        
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(nn.Conv3d(128* red, 64* red, kernel_size=4, stride=1, padding=2)) 
        decoder_layers.append(nn.BatchNorm3d(64* red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(64* red, 64* red, kernel_size=3, stride=1, dropout_rate=drop))

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(nn.Conv3d(64* red, 12, kernel_size=4, stride=1, padding=2))      
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)
        for layer in encoder_layers:
            if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight)

    def encode(self, x):
        x = self.encoder(x)
     #   print(x.shape)
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
       # print(x.shape)
        batch_size = x.size(0)
      #  print(batch_size)
        decoder_reshape_without_batch = self.decoder_reshape[1:]
        x = x.view(batch_size, *decoder_reshape_without_batch)
        x_recon = self.decoder(x)
       # print(x_recon.shape)
        x_recon = F.interpolate(x_recon, size=(64, 64, 2), mode='trilinear', align_corners=False)
        return x_recon

    # Inside the forward method of the VAE class
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        #print(z.shape)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var
    
    def _get_encoder_output_shape(self, x):
        x = self.encoder(x)
        encoder_output_shape = x.view(x.size(0), -1).shape[1]
        decoder_input_shape = x.size()
        return encoder_output_shape, decoder_input_shape

class VAE_Loss(nn.Module):
    def __init__(self, beta):
        super(VAE_Loss, self).__init__()
        self.beta = beta

    def forward(self, decoded, x, mu, log_var):
        reconstruction_loss = F.mse_loss(decoded, x, reduction='sum') / decoded.size(0)
        kl_divergence_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss
        return total_loss, reconstruction_loss, kl_divergence_loss
