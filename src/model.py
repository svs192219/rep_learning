import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

def init_xavier(input_layer, bias_val=0):
    nn.init.xavier_normal_(input_layer.weight);
    nn.init.constant_(input_layer.bias, bias_val);
    return input_layer

class Encoder(nn.Module):  
    def __init__(self, cuda):
        super(Encoder, self).__init__()
        self.cudaPresent = cuda
        self.conv1_1 = init_xavier(nn.Conv2d(3, 8, 5))
        self.bn1_1 = nn.BatchNorm2d(8)
        self.conv1_2 = init_xavier(nn.Conv2d(8, 16, 5))
        self.bn1_2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = init_xavier(nn.Conv2d(16, 32, 3))
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = init_xavier(nn.Conv2d(32, 32, 3))
        self.bn2_2 = nn.BatchNorm2d(32)
        
        self.conv3_1 = init_xavier(nn.Conv2d(32, 64, 3))
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = init_xavier(nn.Conv2d(64, 64, 3))
        self.bn3_2 = nn.BatchNorm2d(64)
        
        self.conv4_1 = init_xavier(nn.Conv2d(64, 128, 3))
        self.bn4_1 = nn.BatchNorm2d(128)
        self.conv4_2 = init_xavier(nn.Conv2d(128, 128, 3))
        self.bn4_2 = nn.BatchNorm2d(128)
        
        self.fc1 = init_xavier(nn.Linear(128 * 12 * 12, 128 * 16))
        self.bn5 = nn.BatchNorm1d(128 * 16)
        self.fc21 = init_xavier(nn.Linear(128 * 16, 256))
        self.fc22 = init_xavier(nn.Linear(128 * 16, 256))

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.pool(F.relu(self.bn1_2(self.conv1_2(x))))
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool(F.relu(self.bn2_2(self.conv2_2(x))))
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.pool(F.relu(self.bn3_2(self.conv3_2(x))))

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.pool(F.relu(self.bn4_2(self.conv4_2(x))))
        
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.bn5(self.fc1(x)))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return self.reparametrize(mu, logvar), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False)
        if self.cudaPresent:
            eps = eps.cuda()
        return (eps * std) + mu

class Decoder(nn.Module):  
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = init_xavier(nn.Linear(256, 64 * 32))
        self.bn1 = nn.BatchNorm1d(64 * 32)
        
        self.fc2 = init_xavier(nn.Linear(64 * 32, 64 * 21 * 21))
        self.bn2 = nn.BatchNorm1d(64 * 21 * 21)
        
        self.dconv1 = init_xavier(nn.ConvTranspose2d(64, 32, 3, stride=3))
        self.bn3 = nn.BatchNorm2d(32)
        
        self.dconv2 = init_xavier(nn.ConvTranspose2d(32, 16, 3, stride=2))
        self.bn4 = nn.BatchNorm2d(16)
        
        self.dconv3 = init_xavier(nn.ConvTranspose2d(16, 3, 4, stride=2))


    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = x.view(-1, 64, 21, 21)
        x = F.relu(self.bn3(self.dconv1(x)))
        x = F.relu(self.bn4(self.dconv2(x)))
        x = F.sigmoid(self.dconv3(x))
        return x

class Discriminator(nn.Module):  
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(256, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

class resnetEncoder(nn.Module):
    def __init__(self, nz, cuda, pretrained=True):
        super(resnetEncoder, self).__init__()
        self.cudaPresent = cuda
        self.nz = nz
        resnet_copy = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(resnet_copy.conv1,
                                      resnet_copy.bn1,
                                      resnet_copy.relu,
                                      resnet_copy.maxpool,
                                      resnet_copy.layer1, 
                                      resnet_copy.layer2,
                                      resnet_copy.layer3,
                                      resnet_copy.layer4,
                                      nn.AvgPool2d(kernel_size=4, stride=4, padding=0, ceil_mode=False, count_include_pad=True)
                                      )
        self.mu     = init_xavier(nn.Linear(512, self.nz))
        self.logvar = init_xavier(nn.Linear(512, self.nz))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return self.reparametrize(mu, logvar), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False)
        if self.cudaPresent:
            eps = eps.cuda()
        return (eps * std) + mu

class dcganDecoder(nn.Module):
    def __init__(self, nz, ngf=8, nc=3):
        super(dcganDecoder, self).__init__()
        self.nz = nz
        self.net = nn.Sequential(
                    # input is Z (batch x nz x 1 x 1), going into a convolution
                    nn.ConvTranspose2d(     nz,   ngf * 32, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 32),
                    nn.ReLU(True),
                    # state size. (ngf*32) x 4 x 4
                    nn.ConvTranspose2d(ngf * 32,  ngf * 16, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 16),
                    nn.ReLU(True),
                    # state size. (ngf*16) x 8 x 8
                    nn.ConvTranspose2d(ngf * 16,   ngf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 16 x 16
                    nn.ConvTranspose2d( ngf * 8,    ngf* 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 32 x 32
                    nn.ConvTranspose2d( ngf * 4,   ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 64 x 64
                    nn.ConvTranspose2d( ngf * 2,       nc, 4, 2, 1, bias=False),
                    #nn.BatchNorm2d(ngf),
                    #nn.ReLU(True),
                    # state size. (ngf) x 128 x 128
                    #nn.ConvTranspose2d(     ngf,        nc, 4, 2, 1, bias=False),
                    nn.Sigmoid()
                    # state size. (nc) x 256 x 256
                    )


    def forward(self, x):
        x = x.view(-1, self.nz, 1, 1)
        x = self.net(x)
        return x
