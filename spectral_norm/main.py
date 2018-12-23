import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model_resnet
import model
from FeedbackModel import FeedbackModel

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append('/home/firas/nets/pytorch-spectral-normalization-gan')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='bce')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--loop_count', type=int, default=2)
parser.add_argument('--train', action='store_true')
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--a1', type=float, default=1)
parser.add_argument('--a2', type=float, default=1)
parser.add_argument('--a3', type=float, default=1)
parser.add_argument('--beta1', type=float, default=0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--disc_iters', type=int, default=5)
parser.add_argument('--discDC', action='store_true')
parser.add_argument('--L1', action='store_true')
parser.add_argument('--dual_input', action='store_true')



args = parser.parse_args()
if args.discDC:
    args.dual_inputs = False

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = args.dim
#number of updates to discriminator for every update to generator 
disc_iters = args.disc_iters

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
if args.model == 'resnet':
    discriminator = model_resnet.Discriminator().cuda()
    generator = model_resnet.Generator(Z_dim, args.a1, args.a2, args.a3).cuda()
else:
    if args.discDC:
        discriminator = model.DiscriminatorDC().cuda()
    else:
        discriminator = model.Discriminator().cuda()
    generator = model.Generator(Z_dim, args.a1, args.a2, args.a3).cuda()
feedbackM = FeedbackModel(netG=generator, netD=discriminator, opt=args)

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(args.beta1,args.beta2))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1,args.beta2))
optim_feedback = optim.Adam(feedbackM.netGA.parameters(), lr=args.lr, betas=(args.beta1,args.beta2))


# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_fb = optim.lr_scheduler.ExponentialLR(optim_feedback, gamma=0.99)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):

        start_time = time.time()
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            optim_feedback.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(feedbackM.gen_result(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(feedbackM.gen_result(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(feedbackM.gen_result(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        optim_feedback.zero_grad()
        res=0
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(feedbackM.gen_result(z)).mean()
        else:
            res = feedbackM.gen_result(z)
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(res), Variable(torch.ones(args.batch_size, 1).cuda()))

        if args.loop_count ==1: # if training base
            gen_loss.backward()
            optim_gen.step()
        else:                   # if training feedback
            if args.L1:
                gen_loss = gen_loss + torch.nn.L1Loss()(res, Variable(feedbackM.fake_B0, requires_grad=False))
            gen_loss.backward()
            optim_feedback.step()

        if batch_idx % 100 == 0:
            print('epoch ', epoch, 'disc loss', disc_loss.item(), 'gen loss', gen_loss.item(), 'time',time.time()-start_time)
    scheduler_d.step()
    scheduler_g.step()
    scheduler_fb.step()

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def evaluate(epoch, dir):

    samples = feedbackM.gen_result(fixed_z).cpu().data.numpy()[:64]


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)
    out_path = dir + 'out/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    plt.savefig(out_path + '/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

#os.makedirs(args.checkpoint_dir), exist_ok=True)

def load_models(args, loadFB = True):
    generator.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.load_epoch))))
    discriminator.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'disc_{}'.format(args.load_epoch))))
    if loadFB:
        feedbackM.netGA.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'fb_{}'.format(args.load_epoch))))

def eval_model(args):
    ut.get_inception_score(feedbackM.gen_result, args.checkpoint_dir, 'eval%d'%feedbackM.loop_count )
    import tflib.inception_score
    if True:
        ut.save_intermediate_res(feedbackM.get_intermediate_fake(), args.checkpoint_dir, 'fake')
        inception_score = ut.inception_score_from_file(args.checkpoint_dir, 'eval%d'%feedbackM.loop_count)
        print('inception score: %f , %f' % (inception_score[0], inception_score[1]))

import tflib as lib
import utils as ut
if args.train:
    start_epoch = 0
    if args.loop_count > 1:
        load_models(args,False)
        start_epoch = args.load_epoch

    for epoch in range(start_epoch,start_epoch + args.epochs):
        train(epoch)
        evaluate(epoch,args.checkpoint_dir )
        if epoch % 20 == 19:
            torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
            torch.save(feedbackM.netGA.state_dict(), os.path.join(args.checkpoint_dir, 'fb_{}'.format(epoch)))
    eval_model(args)
else:
    load_models(args,(args.loop_count>1))
    eval_model(args)
    # ut.get_inception_score(generator, args.checkpoint_dir, 'eval')
    # import tflib.inception_score
    #
    # inception_score = ut.inception_score_from_file(args.checkpoint_dir, 'eval')
    # print('inception score: %f , %f' % (inception_score[0], inception_score[1]))
