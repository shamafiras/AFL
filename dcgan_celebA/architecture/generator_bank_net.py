from torch import nn
import torch
from architecture.main_net import MainNet
from architecture.FeedbackModel import GeneratorAFL


class GeneratorBankNet(nn.Module):
    def __init__(self, Disc, opt):
        super(GeneratorBankNet, self).__init__()
        self.main = MainNet()
        self.netGA = GeneratorAFL()
        self.netD = Disc
        self.alpha1 = opt.a1
        self.alpha2 = opt.a2
        self.alpha3 = opt.a3
        self.alpha4 = opt.a4
        self.loop_count = opt.loop_count
        self.opt = opt

    def _forward_afl(self, x):
        out = self.main.layer0(x)
        out = self.main.layer1(out + self.alpha1 * self.netGA.trans_block0(torch.cat([self.netGA.feedback0, out], 1)))
        out = self.main.layer2(out + self.alpha2 * self.netGA.trans_block1(torch.cat([self.netGA.feedback1, out], 1)))
        out = self.main.layer3(out + self.alpha3 * self.netGA.trans_block2(torch.cat([self.netGA.feedback2, out], 1)))
        out = self.main.layer4(out + self.alpha4 * self.netGA.trans_block3(torch.cat([self.netGA.feedback3, out], 1)))
        return out

    def forward(self, x):

        gen_output = self.main(x)
        d_out = self.netD(gen_output)

        setattr(self, 'fake_B0', gen_output)

        for step in range(1, self.loop_count):

            self.netGA.set_input_disc(self.netD.getLayersOutDet())
            gen_output = self._forward_afl(x)
            d_out = self.netD(gen_output)

            setattr(self, 'fake_B%s_score' % step, d_out)
            setattr(self, 'fake_B%s' % step, gen_output)

        return gen_output
