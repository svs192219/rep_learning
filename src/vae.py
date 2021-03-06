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
from tensorboardX import SummaryWriter

from model import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def load_data(opts):

    transform = transforms.Compose(
                          [transforms.Resize((128, 128)),
                           transforms.ToTensor(), 
                           transforms.Normalize(mean = opts.mean, std = opts.std)])

    dataset = ImageFolder(root=opts.dataset_path, 
                           transform=transform)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size,
                             shuffle=True, num_workers=opts.num_workers)

    return dataloader

def repeat_vector(vec, dims):
    return torch.Tensor(vec).view(1, 3, 1, 1).repeat(dims[0], 1, dims[2], dims[3])

def unnormalize(sample):
    return (sample * repeat_vector(opts.std, sample.size())) + repeat_vector(opts.mean, sample.size())


def train(opts):
    encoder = resnetEncoder(nz=opts.latent_dim, cuda=opts.cuda)
    decoder = dcganDecoder(nz=opts.latent_dim)
    if opts.cuda:  
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    dataloader = load_data(opts)

    enc_optimizer = optim.Adam(encoder.parameters(), lr=opts.lr)
    if isinstance(encoder, resnetEncoder):
        enc_params = [{'params': encoder.features.parameters(), 'lr': opts.lr * 0.1},
                {'params': encoder.mu.parameters()},
                {'params': encoder.logvar.parameters()}]
        enc_optimizer = optim.Adam(enc_params, lr=opts.lr)
    
    dec_optimizer = optim.Adam(decoder.parameters(), lr=opts.lr)

    writer = SummaryWriter(opts.save_path)

    writer.add_text('Training options', json.dumps(vars(opts), indent=4), 0)

    iters = 0
    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        # exp_lr_scheduler.step()

        running_loss = 0
        for batch_idx, data in enumerate(dataloader):
            # switch models to training mode
            encoder.train()
            decoder.train()

            # get the inputs
            X, _ = data
            # wrap them in Variable
            X = Variable(X)
            # convert to cuda if available
            if opts.cuda:
                X = X.cuda()

            # zero the parameter gradients
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            
            z_sample, z_mu, z_logvar = encoder(X)
            X_sample = decoder(z_sample)

            def loss_func(X_sample, mu, logvar):
                # reconstruction loss
                recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False)

                # KL divergence loss
                kld_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

                return (recon_loss + kld_loss)

            overall_loss = loss_func(X_sample, z_mu, z_logvar)
            # backpropagate
            overall_loss.backward()
            running_loss += overall_loss.item()
            dec_optimizer.step()
            enc_optimizer.step()

            iters += 1

            if batch_idx % opts.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * X.size(0), len(dataloader.dataset),
                    overall_loss.item() / X.size(0)))

                # reconstructed image
                encoder.eval()
                decoder.eval()
                z_sample, mu, logvar = encoder(X)
                X_sample = decoder(z_sample)
                
                comparison = torch.cat([X, X_sample])
                # img_name = 'results/reconstruction_' + str(batch_idx) + '_' + str(epoch) + '.png'
                # utils.save_image(comparison.cpu().data,
                        # os.path.join(opts.save_path, img_name), nrow=n)
                grid = utils.make_grid(unnormalize(comparison.cpu().data), nrow=opts.batch_size)
                writer.add_image('image_reconstruction', grid, iters)
                writer.add_scalar('loss', overall_loss.item(), iters)
                
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, running_loss / len(dataloader.dataset)))

        torch.save(encoder.state_dict(), os.path.join(opts.save_path, 'encoder'))
        torch.save(decoder.state_dict(), os.path.join(opts.save_path, 'decoder'))
        encoder.eval()
        decoder.eval()
        sample = Variable(torch.randn(16, opts.latent_dim))

        if opts.cuda:
            sample = sample.cuda()
        sample = decoder(sample).cpu().data
        sample = unnormalize(sample)
        # img_name = 'results/sample_' + str(epoch) + '_' + str(batch_idx) + '.png'
        # utils.save_image(sample.data.view(16, 3, 256, 256),
                   # os.path.join(opts.save_path, img_name))
        writer.add_image('random_sample', utils.make_grid(sample), (epoch + 1))


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
    parser.add_argument('--log_interval', type=int, default=30)
    parser.add_argument('--latent_dim', type=int, default=64)

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    json.dump(vars(opts), open(os.path.join(opts.save_path, 'opts.json'), 'w'))
    train(opts)
