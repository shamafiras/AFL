import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_size=3, ndf=128):
        super(Discriminator, self).__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.l1 = nn.Sequential(# input size is in_size x 64 x 64
            nn.Conv2d(self.in_size, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(# state size: ndf x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),)
        self.l3 = nn.Sequential(# state size: (ndf * 2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),)
        self.l4 = nn.Sequential(# state size: (ndf * 4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),)
        self.l5 = nn.Sequential(# state size: (ndf * 8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        self.l1out = self.l1(input)
        self.l2out = self.l2(self.l1out)
        self.l3out = self.l3(self.l2out)
        self.l4out = self.l4(self.l3out)
        return self.l5(self.l4out)


    ## USED FOR FEEDBACK LOOP
    def getLayersOutDet(self):
        return [self.l1out.detach(), self.l2out.detach(), self.l3out.detach(), self.l4out.detach()]