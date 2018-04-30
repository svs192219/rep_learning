from __future__ import print_function, division

import os
import sys
import json
import pickle
import argparse

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets

from model import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def load_data(opts):

    transform = transforms.Compose(
                          [transforms.Scale((256, 256)),
                           transforms.ToTensor(), 
                           transforms.Normalize(mean = opts.mean, std = opts.std)])

    dataset = ImageFolder(root=opts.dataset_path, 
                           transform=transform)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size,
                             shuffle=True, num_workers=opts.num_workers)

    return dataloader

def train(opts):
    encoder = Encoder()
    decoder = Decoder()
    # discriminator = Discriminator()
    if torch.cuda.is_available():  
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        # discriminator = discriminator.cuda()
    # Set learning rates
    gen_lr, reg_lr = 0.0006, 0.0008
    
    # Set optimizators
    dec_optimizer = optim.Adam(decoder.parameters(), lr=gen_lr)  
    enc_optimizer = optim.Adam(encoder.parameters(), lr=gen_lr)  
    gen_optimizer = optim.Adam(encoder.parameters(), lr=reg_lr)  
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=reg_lr) 

    dataloader = load_data(opts)

    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        # exp_lr_scheduler.step()
            
        print('Running epoch %d' % epoch)
        for i, data in enumerate(dataloader):

            if i%15 == 0:
                print('Batch id : %d' % i)

            # get the inputs
            X, Y = data
            # wrap them in Variable
            X, Y = Variable(X), Variable(Y)
            # convert to cuda if available
            if opts.cuda:
                X = X.cuda()
                Y = Y.cuda()

            # zero the parameter gradients
            dec_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            # reconstruction loss 
            z_sample = encoder(X)
            X_sample = decoder(z_sample)
            recon_loss = F.binary_cross_entropy(X_sample, X)
            recon_loss.backward()
            dec_optimizer.step()
            enc_optimizer.step()

            # generate fake samples from encoder
            encoder.eval()
            z_real_gauss = Variable(torch.randn(X.size(0), 256))
            if torch.cuda.is_available():
                z_real_gauss = z_real_gauss.cuda()
            z_fake_gauss = encoder(X)

            # train discriminator
            discriminator.train()
            D_real_gauss, D_fake_gauss = discriminator(z_real_gauss), discriminator(z_fake_gauss)
            D_loss = -torch.mean(torch.log(D_real_gauss) + torch.log(1 - D_fake_gauss))
            D_loss.backward()       # Backpropagate loss
            disc_optimizer.step()   # Apply optimization step

            # train generator
            encoder.train()   # Back to use dropout  
            z_fake_gauss = encoder(X)
            D_fake_gauss = discriminator(z_fake_gauss)

            G_loss = -torch.mean(torch.log(D_fake_gauss))  
            G_loss.backward()
            gen_optimizer.step()

            if i % 15 == 0:
                sample = Variable(torch.randn(16, 256))
                if opts.cuda:
                    sample = sample.cuda()
                sample = decoder(sample).cpu()
                sample = (sample * opts.std) + opts.mean
                img_name = 'results/sample_' + str(epoch) + '_' + str(i) + '.png'
                utils.save_image(sample.data.view(16, 3, 256, 256),
                           os.path.join(opts.save_path, img_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='clock_data/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--lr_step_size', type=int, default=10, help='step size for LR scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scale', type=float, default=0.1)

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    json.dump(vars(opts), open(os.path.join(opts.save_path, 'opts.json'), 'w'))
    train(opts)