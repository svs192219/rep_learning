import torch
import torch.nn as nn
import torch.nn.functional as F

def init_xavier(input_layer, bias_val=0):
    nn.init.xavier_normal(input_layer.weight);
    nn.init.constant(input_layer.bias, bias_val);
    return input_layer

class Encoder(nn.Module):  
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_1 = init_xavier(nn.Conv2d(3, 8, 5))
        self.conv1_2 = init_xavier(nn.Conv2d(8, 16, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = init_xavier(nn.Conv2d(16, 32, 3))
        self.conv2_2 = init_xavier(nn.Conv2d(32, 32, 3))
        self.conv3_1 = init_xavier(nn.Conv2d(32, 64, 3))
        self.conv3_2 = init_xavier(nn.Conv2d(64, 64, 3))
        self.conv4_1 = init_xavier(nn.Conv2d(64, 128, 3))
        self.conv4_2 = init_xavier(nn.Conv2d(128, 128, 3))
        self.fc1 = init_xavier(nn.Linear(128 * 12 * 12, 128 * 16))
        self.fc21 = init_xavier(nn.Linear(128 * 16, 256))
        self.fc22 = init_xavier(nn.Linear(128 * 16, 256))

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = self.pool(F.relu(self.conv3_2(x)))
        x = F.relu(self.conv4_1(x))
        x = self.pool(F.relu(self.conv4_2(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return self.reparametrize(mu, logvar), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

class Decoder(nn.Module):  
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = init_xavier(nn.Linear(256, 64 * 32))
        self.fc2 = init_xavier(nn.Linear(64 * 32, 64 * 21 * 21))
        self.dconv1 = init_xavier(nn.ConvTranspose2d(64, 32, 3, stride=3))
        self.dconv2 = init_xavier(nn.ConvTranspose2d(32, 16, 3, stride=2))
        self.dconv3 = init_xavier(nn.ConvTranspose2d(16, 3, 4, stride=2))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 21, 21)
        x = F.relu(self.dconv1(x))
        x = F.relu(self.dconv2(x))
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