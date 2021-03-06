import os, sys
sys.path.append(os.getcwd())

import time
import functools
import argparse

import numpy as np
#import sklearn.datasets

import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

import pdb
import gpustat

# import models.dcgan as dcgan
from wgan import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer

import torch.nn.init as init

import json
import pickle

import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets
from tensorboardX import SummaryWriter


RESTORE_MODE = False # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration 
MODE = 'wgan-gp' # dcgan, wgan
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 8# Batch size. Must be a multiple of N_GPUS
END_ITER = 100000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each image
OLDGAN = False


def showMemoryUsage(device=1):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def load_data(path_to_folder, classes):
    data_transform = transforms.Compose([
                 transforms.Resize((64, 64)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
    
    dataset = datasets.ImageFolder(root=(path_to_folder + classes), transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=5, drop_last=True, pin_memory=True)
    return dataset_loader

def training_data_loader(opts):
    return load_data(opts.dataset_path, '/train') 

def val_data_loader():
    return load_data(opts.dataset_path, '/val') 

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(netG, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
    	noisev = noise 
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 3, 64, 64)
    samples = samples * 0.5 + 0.5
    return samples

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


def gen_rand_noise():
    if OLDGAN:
        noise = torch.FloatTensor(BATCH_SIZE,128,1,1)
        noise.resize_(BATCH_SIZE,128,1,1).normal_(0,1)
    else:
        noise = torch.randn(BATCH_SIZE, 128)
        noise = noise.to(device)

    return noise



#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train(opts):
        
    fixed_noise = gen_rand_noise()


    if RESTORE_MODE:
        aG = torch.load(opts.save_path + "generator.pt")
        aD = torch.load(opts.save_path + "discriminator.pt")
    else:
        if MODE == 'wgan-gp':
            aG = GoodGenerator(64,64*64*3)
            aD = GoodDiscriminator(64)
            OLDGAN = False
        # elif MODE == 'dcgan':
        #     aG = FCGenerator()
        #     aD = DCGANDiscriminator()
        #     OLDGAN = False
        # else:
        #     aG = dcgan.DCGAN_G(DIM, 128, 3, 64, 1, 0)
        #     aD = dcgan.DCGAN_D(DIM, 128, 3, 64, 1, 0)
        #     OLDGAN= True
        
        aG.apply(weights_init)
        aD.apply(weights_init)

    LR = 1e-4
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0,0.9))
    one = torch.FloatTensor([1])
    mone = one * -1
    aG = aG.to(device)
    aD = aD.to(device)
    one = one.to(device)
    mone = mone.to(device)

    writer = SummaryWriter()

    dataloader = training_data_loader(opts) 
    dataiter = iter(dataloader)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad_(True)
            fake_data = aG(noise)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        end = timer()
        print('---train G elapsed time: %f' % (end - start))
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise()
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            end = timer(); print('---gen G elapsed time: %f' % (end-start))
            start = timer()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading
            end = timer(); print('---load real imgs elapsed time: %f' % (end-start))
            start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1 and not OLDGAN:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                #writer.add_scalar('data/disc_fake', disc_fake, iteration)
                #writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                #writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/fake_data_mean', fake_data.mean())
                #writer.add_scalar('data/real_data_mean', real_data.mean())
                #if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
                if iteration %200==199:
                    body_model = [i for i in aD.children()][0]
                    layer1 = body_model.conv
                    xyz = layer1.weight.data.clone()
                    tensor = xyz.cpu()
                    tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                    writer.add_image('D/conv1', tensors, iteration)

            end = timer(); print('---train D elapsed time: %f' % (end-start))
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(opts.save_path + 'time', time.time() - start_time)
        lib.plot.plot(opts.save_path + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(opts.save_path + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(opts.save_path + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 200 == 199:
            val_loader = val_data_loader() 
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
               	imgs = imgs.to(device)
                with torch.no_grad():
            	    imgs_v = imgs

                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(opts.save_path + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()	
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, opts.save_path + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
	#----------------------Save model----------------------
            torch.save(aG, opts.save_path + "generator.pt")
            torch.save(aD, opts.save_path + "discriminator.pt")
        lib.plot.tick()


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

    opts = parser.parse_args()
    opts.mean = [0.485, 0.456, 0.406]
    opts.std = [0.229, 0.224, 0.225]

    json.dump(vars(opts), open(os.path.join(opts.save_path, 'opts.json'), 'w'))
    train(opts)

