# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm

channels = 3
leak = 0.1
w_g = 4

class Generator(nn.Module):
    def __init__(self, z_dim, a1, a2, a3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.alpha1 = a1
        self.alpha2 = a2
        self.alpha3 = a3

        self.b0 = nn.Sequential(nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
                      nn.BatchNorm2d(512),
                      nn.ReLU())
        self.b1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)),
                    nn.BatchNorm2d(256),
                    nn.ReLU())
        self.b2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
        self.b3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.b4 = nn.Sequential(nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1, 1)),
                                    nn.Tanh())
        self.model = nn.Sequential(self.b0, self.b1, self.b2, self.b3, self.b4)

    def _forward(self, z):
        self.l1out = self.b0(z.view(-1, self.z_dim, 1, 1))
        self.l2out = self.b1(self.l1out)
        self.l3out = self.b2(self.l2out)
        output = self.b4(self.b3(self.l3out))
        return output

    def forward(self, z, feedback_layers=None):
        if feedback_layers == None:
            return self._forward(z)
        else:
            self.l1out = self.b0(z.view(-1, self.z_dim, 1, 1))
            self.l2out = self.b1(self.l1out + self.alpha1*feedback_layers[0])
            self.l3out = self.b2(self.l2out + self.alpha2*feedback_layers[1])
            output = self.b3(self.l3out + self.alpha3*feedback_layers[2])
            return self.b4(output)
    def getLayersOutDet(self):
        return [self.l1out.detach(), self.l2out.detach(), self.l3out.detach()]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        self.l1out = m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        self.l2out = m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        self.l3out = m = nn.LeakyReLU(leak)(self.conv7(m))

        return self.fc(m.view(-1,w_g * w_g * 512))
    ## ADDED DUE TO FEEDBACK LOOP
    def getLayersOutDet(self):
        return [self.l1out.detach(), self.l2out.detach(), self.l3out.detach()]

class DiscriminatorDC(Discriminator):
    def __init__(self):
        super(DiscriminatorDC, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(channels, 64, 3, stride=1, padding=(1, 1)),nn.BatchNorm2d(64))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)), nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)), nn.BatchNorm2d(128))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)), nn.BatchNorm2d(128))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)), nn.BatchNorm2d(256))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)), nn.BatchNorm2d(256))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)), nn.BatchNorm2d(512))

        self.fc = nn.Linear(w_g * w_g * 512, 1)



