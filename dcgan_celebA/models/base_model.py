from tqdm import tqdm
import torch
from torchvision import transforms
import torch.nn as nn
import os
import torchvision
import matplotlib.pyplot as plt
import src.utils as utils


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        utils.print_options(opt)
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.train_main_loader = utils.get_data_loader(opt, train=True, main=True)
        self.train_bank_loader = utils.get_data_loader(opt, train=True, main=False)
        self.train_loader = self.train_main_loader
        self.main_gen_optimizer = None
        self.bank_gen_optimizer = None
        self.main_disc_optimizer = None
        #self.bank_disc_optimizer = None
        self.net = None
        self.disc = None
        self.eval_tensor = torch.randn((opt.eval_noise_batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.device)

    def train(self, main_training=True):
        criterion = nn.BCELoss().to(self.device)
        if main_training:
            gen_optimizer = self.main_gen_optimizer
            disc_optimizer = self.main_disc_optimizer
            num_of_epochs = self.opt.main_epochs
            self.disc = self.main_disc
            self.train_loader = self.train_main_loader
        else:
            gen_optimizer = self.bank_gen_optimizer
            disc_optimizer = self.main_disc_optimizer
            num_of_epochs = self.opt.bank_epochs
            self.disc = self.main_disc
            self.train_loader = self.train_bank_loader
            for parm in self.net.main.parameters():
                parm.requires_grad = False
        for epoch in range(num_of_epochs):
            self.net.train()
            self.disc.train()
            iter_count = 0
            for batch_id, input_batch in tqdm(enumerate(self.train_loader)):
                current_batch_size = input_batch.shape[0]
                iter_count += current_batch_size
                input_batch = input_batch.to(self.device)

                # train discriminator
                disc_optimizer.zero_grad()
                disc_real_result = self.disc(input_batch).squeeze()
                label_real = torch.ones(current_batch_size).to(self.device)
                disc_real_loss = criterion(disc_real_result, label_real)
                label_fake = torch.zeros(current_batch_size).to(self.device)
                noise = torch.randn((current_batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.device)
                gen_result = self.forward(noise)
                disc_fake_result = self.disc(gen_result.detach()).squeeze()
                disc_fake_loss = criterion(disc_fake_result, label_fake)
                disc_train_loss = disc_real_loss + disc_fake_loss
                disc_train_loss.backward()
                disc_optimizer.step()

                # train generator
                gen_optimizer.zero_grad()
                noise = torch.randn((current_batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.device)
                gen_result = self.forward(noise)
                disc_fake_result = self.disc(gen_result).squeeze()
                gen_train_loss = criterion(disc_fake_result, label_real)
                gen_train_loss.backward()
                gen_optimizer.step()

                if (batch_id + 1) % self.opt.eval_iter == 0:
                    print('\n===============epoch: %d, iter: %d/%d===============' % (epoch, (batch_id + 1), len(self.train_loader)))
                    print('gen loss: %.5f, disc real loss: %.5f, disc fake loss: %.5f' % (gen_train_loss.item(), disc_real_loss.item(), disc_fake_loss.item()))
                if (batch_id + 1) % self.opt.intermediate_images_iter == 0:
                    self.show_intermediate(gen_result)
                if (batch_id + 1) % self.opt.save_image_iter == 0:
                    self.save_evaluation_images(epoch, alpha=int(not main_training))

            self.save_nets(epoch, main_training=main_training, latest=(epoch + 1 == num_of_epochs))

    def save_nets(self, epoch, main_training=True, latest=False):
        model_name = 'main' if main_training else 'bank'
        if not latest:
            model_name = '%s_epoch_%d.pth' % (model_name, epoch)
            path = self.opt.model_save_dir
        else:
            model_name = '%s_latest.pth' % model_name
            path = self.opt.model_save_dir
        torch.save(self.net.cpu().state_dict(), os.path.join(path, 'gen_%s' % model_name))
        torch.save(self.disc.cpu().state_dict(), os.path.join(path, 'disc_%s' % model_name))
        print('saved %s to %s' % (('gen_%s' % model_name), path))
        if main_training and latest:
            torch.save(self.net.main.cpu().state_dict(), os.path.join(path, 'original_%s' % model_name))
            print('saved %s to %s' % ('original_%s' % model_name, path))
        self.net.to(self.device)
        self.disc.to(self.device)

    def forward(self, input_batch):
        return self.net(input_batch)

    def show_intermediate(self, gen_result):
        image = torchvision.utils.make_grid(self.recover_tensor(gen_result).clamp(min=0.0, max=1), nrow=16)
        image = transforms.ToPILImage()(image.cpu())
        plt.clf()
        plt.imshow(image)
        plt.pause(.001)

    def save_evaluation_images(self, epoch, alpha=0, latest=False):
        output_tensor = self.recover_tensor(self.forward(self.eval_tensor, alpha=alpha).squeeze(dim=0)).clamp(min=0.0, max=1).cpu()
        if latest:
            save_name = 'latest_alpha_%.3f.jpeg' % alpha
        else:
            save_name = 'epoch_%d_alpha_%.3f.jpeg' % (epoch, alpha)
        save_path = os.path.join(self.opt.images_save_dir, save_name)
        image = torchvision.utils.make_grid(output_tensor, nrow=16)
        utils.save_tensor_as_image(save_path, image)

    def recover_tensor(self, image_tensor):
        mean = image_tensor.new_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        std = image_tensor.new_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        return (image_tensor * std) + mean

    def init_paths(self):
        utils.make_dirs(self.opt.model_save_dir)
        utils.make_dirs(self.opt.images_save_dir)

    def write_config(self):
        with open(os.path.join(self.opt.main_dir_name, 'config.txt'), 'w') as f:
            f.write(str(vars(self.opt)))
