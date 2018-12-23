import torch
from torch import nn

class TransBlock(nn.Module):
    def __init__(self, size):
        super(TransBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(size, size, 3, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.Conv2d(size, size, 3, padding=1),
            nn.BatchNorm2d(size))

    def forward(self, input):
        return self.main(input)

class TransBlockRes(nn.Module):
    def __init__(self, size):
        super(TransBlockRes, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2*size, size, 3, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.Conv2d(size, size, 3, padding=1),
            nn.BatchNorm2d(size))

    def forward(self, input):
        return self.main(input)

class TransBlock2(nn.Module):
    def __init__(self, size):
        super(TransBlock2, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(size, size, 3),
            nn.BatchNorm2d(size),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(size, size, 3),
            nn.BatchNorm2d(size))

    def forward(self, input):
        return self.main(input)

class GeneratorAdd(nn.Module):
    def __init__(self, opt, ):
        super(GeneratorAdd, self).__init__()
        if opt.dual_input:
            self.trans_block1 = TransBlockRes(4*opt.dim)
            self.trans_block2 = TransBlockRes(2*opt.dim)
            self.trans_block3 = TransBlockRes(1*opt.dim)
        else:
            self.trans_block1 = TransBlock(4 * opt.dim)
            self.trans_block2 = TransBlock(2 * opt.dim)
            self.trans_block3 = TransBlock(1 * opt.dim)

    def set_feedback(self, layers_input):
        self.feedback1 = layers_input[2]
        self.feedback2 = layers_input[1]
        self.feedback3 = layers_input[0]

    def set_orig(self, layers_input):
        self.orig1 = layers_input[0]
        self.orig2 = layers_input[1]
        self.orig3 = layers_input[2]

    def forward(self):
        if isinstance(self.trans_block1, TransBlockRes):
            out1 = self.trans_block1(torch.cat([self.feedback1, self.orig1], 1))
            out2 = self.trans_block2(torch.cat([self.feedback2, self.orig2], 1))
            out3 = self.trans_block3(torch.cat([self.feedback3, self.orig3], 1))
            return [out1, out2, out3]
        else:
            return [self.trans_block1(self.feedback1), self.trans_block2(self.feedback2), self.trans_block3(self.feedback3)]

class ResnetGeneratorAdd(GeneratorAdd):
    def __init__(self, opt):
        super(ResnetGeneratorAdd, self).__init__(opt)
        self.trans_block1 = TransBlockRes(opt.dim)
        self.trans_block2 = TransBlockRes(opt.dim)
        self.trans_block3 = TransBlockRes(opt.dim)

    def forward(self):
        if isinstance(self.trans_block1, TransBlockRes):
            out1 = None
            out2 = self.trans_block2(torch.cat([self.feedback2, self.orig2], 1))
            out3 = self.trans_block3(torch.cat([self.feedback3, self.orig3], 1))
            return [out1, out2, out3]
        else:
            return [None, self.trans_block2(self.feedback2),
                    self.trans_block3(self.feedback3)]

class FeedbackModel:
    def __init__(self, netG, netD, opt):
        self.netG = netG
        self.netD = netD
        self.opt = opt
        self.z_dim = opt.dim
        if opt.model =='dcgan':
            self.netGA = GeneratorAdd(opt).cuda()
        elif opt.model =='resnet':
            self.netGA = ResnetGeneratorAdd(opt).cuda()
        else:
            raise NotImplemented('model name mismatch')
        self.loop_count = opt.loop_count

    def resForward(self, z):
        self.l1out = self.netG.b0(z.view(-1, self.z_dim, 1, 1))
        self.l2out = self.netG.b1(self.l1out + self.netG.alpha1 * self.netGA.trans_block1(torch.cat([self.netGA.feedback1,self.l1out], 1)))
        self.l3out = self.netG.b2(self.l2out + self.netG.alpha2 * self.netGA.trans_block2(torch.cat([self.netGA.feedback2,self.l2out], 1)))
        output = self.netG.b3(self.l3out + self.netG.alpha3 * self.netGA.trans_block3(torch.cat([self.netGA.feedback3,self.l3out], 1)))
        return self.netG.b4(output)

    def gen_result(self, x):
        gen_output = self.netG(x)
        d_out = self.netD(gen_output)#.mean()
        setattr(self, 'fake_B0', gen_output)

        for step in range(1, self.loop_count):
            self.netGA.set_feedback(self.netD.getLayersOutDet())
            if self.opt.dual_input:
                gen_output = self.resForward(x)
            else:
                self.netGA.set_orig(self.netG.getLayersOutDet())
                feedback = self.netGA()
                gen_output = self.netG(x, feedback)
            d_out = self.netD(gen_output)#.mean()

            setattr(self, 'fake_B%s_score' % step, d_out)
            setattr(self, 'fake_B%s' % step, gen_output)

        return gen_output

    def get_intermediate_fake(self):
        inter_fakes = []
        for iter in range(self.loop_count):
            inter_fakes.append(getattr(self, 'fake_B%s' % iter))
        return inter_fakes

# EXAMPLES OF IMPLEMENTATION IN DISCRIMANATOR & GENERATOR

    # def getLayersOutDet(self):
    #     return [self.layer1out.detach(), self.layer2out.detach(), self.layer3out.detach()]