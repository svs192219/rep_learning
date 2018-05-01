import torch
import torch.nn as nn
import torch.nn.functional as F
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
