# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:34:37 2021

@author: joser
"""

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

class VAE(nn.Module):
    
    def __init__(self, 
                 n_channel,
                 z_dim,
                 hdims,
                 beta=1.0,
                 lr=1e-4,
                 lr_decay=0.98,
                 kl_tol=0.0,
                 sampling=1):
        
        super().__init__()
        
        self.n_channel = n_channel
        self.z_dim= z_dim
        self.hdims = hdims.copy()
        self.beta = beta
        self.lr = lr
        self.lr_decay = lr_decay
        self.kl_tol = kl_tol
        self.sampling = sampling
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for hdim in hdims:
            self.encoder.append(nn.Sequential(nn.Conv2d(n_channel,
                                                        hdim,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=1),
                                              nn.ReLU()))
            n_channel = hdim
            
        self.fc1 = nn.Linear(hdims[-1]*4, 128)
            
        self.mu = nn.Linear(128, z_dim).to(self.device)
        self.logvar = nn.Linear(128, z_dim).to(self.device)
        
        hdims.reverse()
        
        self.decoder_input = nn.Linear(z_dim, self.hdims[-1]*4).to(self.device)
        
        for i in range(len(hdims)-1):
            self.decoder.append(nn.Sequential(nn.ConvTranspose2d(hdims[i], 
                                                                  hdims[i+1],
                                                                  kernel_size=4,
                                                                  stride=2,
                                                                  padding=1),
                                              nn.ReLU()))
            
        self.final_layer=nn.Sequential(
                            nn.ConvTranspose2d(hdims[-1],
                                                self.n_channel,
                                                kernel_size=4,
                                                stride=2,
                                                padding=1),
                            nn.Sigmoid())
        
        
        # trainnig proc.
        self.bce = nn.BCELoss(reduction="sum")
        
        
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return (self.decode(z), mu, logvar)
    
    def encode(self, x):
        
        for i, module in enumerate(self.encoder):
            x = module(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder_input(z).view(-1, self.hdims[-1], 2, 2)
        for i, module in enumerate(self.decoder):    
            x = module(x)
        x = self.final_layer(x)
        return x
    
    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon
        if self.sampling > 1:
            z_ = z.mean(dim=1)
        return z
    
    def generate(self, x):
        return self.forward(x)[0]
    
    def compute_loss(self, input, dec_outp, mu, logvar):
        # rec_loss = F.mse_loss(dec_outp, input)
        rec_loss = self.bce(dec_outp, input)
        KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp())
        return rec_loss + self.beta*KL_loss
    
    
if __name__ == "__main__":
    
    vae = VAE(1, 16, [8, 16, 32, 64]).to("cuda:0")
    
    input = torch.rand((10, 1, 32, 32)).to("cuda:0")
    
    recons, mu, logvar = vae(input)
    
    loss = vae.compute_loss(input, recons, mu, logvar)
    
    
    
    
    
        
        
        
        
        
        