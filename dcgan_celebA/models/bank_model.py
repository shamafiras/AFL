import torch
from models.base_model import BaseModel
from architecture.generator_bank_net import GeneratorBankNet
from architecture.discriminator import Discriminator
# from arcitecture.transformer_net import TransformerNetBank
from torch.optim import Adam
import src.utils as utils
import os


class BankModel(BaseModel):
    def __init__(self, opt):
        super(BankModel, self).__init__(opt)
        self.main_disc = Discriminator()
        self.net = GeneratorBankNet(self.main_disc, opt)

        self.main_gen_optimizer = Adam(self.net.main.parameters(), lr=opt.gen_learning_rate_main, betas=(0.5, 0.999))
        self.bank_gen_optimizer = Adam(self.net.netGA.parameters(), lr=opt.gen_learning_rate_main, betas=(0.5, 0.999))
        self.main_disc_optimizer = Adam(self.main_disc.parameters(), lr=opt.disc_learning_rate_main, betas=(0.5, 0.999))

        self.net.to(self.device)
        self.main_disc.to(self.device)

    def load_pre_trained(self):
        if self.opt.pre_trained_model == 'none':
            print('No pre trained model')
        self.net.main.load_state_dict(torch.load(self.opt.pre_trained_model))
        print('%s pre trained model loaded' % self.opt.pre_trained_model)
        self.net.to(self.device)
        if self.opt.pre_disc_trained_model != 'none':
            self.main_disc.load_state_dict(torch.load(self.opt.pre_disc_trained_model))
            print('%s pre trained disc model loaded' % self.opt.pre_disc_trained_model)
        self.main_disc.to(self.device)
