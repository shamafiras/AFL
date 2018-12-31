from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QLabel, QApplication, QPushButton, QInputDialog
from PySide2 import QtGui, QtCore
from skimage import io
from gui.base_widget import BaseWidget
import os
from models.bank_model import BankModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
import src.utils as utils
from PIL import Image


class DcGanWidget(BaseWidget):
    def __init__(self, app_name='Dc Gan Widget', opt=None, manual_seed=None):
        super(DcGanWidget, self).__init__(app_name=app_name)
        self.opt = opt
        # make main layout
        self.main_layout = QHBoxLayout()
        self.generated_image_label = QLabel(self)
        generated_image_layout = self.make_image_layout(self.generated_image_label, 'Generated Images')
        self.generated_image_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.main_layout.addLayout(generated_image_layout)
        # make alphas sliders
        self.alpha1 = 0
        alpha_slider_layout1, self.alpha_slider1, self.alpha_txt1 = self.make_slider_layout(self.alpha_slider_changed,
                                                                                        val=self.alpha1)
        self.alpha_txt1.setText("alpha = %.3f" % self.alpha1)

        self.alpha2 = 0
        alpha_slider_layout2, self.alpha_slider2, self.alpha_txt2 = self.make_slider_layout(self.alpha_slider_changed,
                                                                                         val=self.alpha2)
        self.alpha_txt2.setText("alpha = %.3f" % self.alpha2)
        self.alpha3 = 0
        alpha_slider_layout3, self.alpha_slider3, self.alpha_txt3 = self.make_slider_layout(self.alpha_slider_changed,
                                                                                         val=self.alpha3)
        self.alpha_txt3.setText("alpha = %.3f" % self.alpha3)

        self.alpha4 = 0
        alpha_slider_layout4, self.alpha_slider4, self.alpha_txt4 = self.make_slider_layout(self.alpha_slider_changed,
                                                                                         val=self.alpha4)
        self.alpha_txt4.setText("alpha = %.3f" % self.alpha4)

        # make im_num_to_save slider
        self.im_num_to_save = 1
        im_num_to_save_slider_layout, self.im_num_to_save_slider, self.im_num_to_save_txt = self.make_slider_layout(self.im_num_to_save_slider_changed,
                                                                                            val=self.im_num_to_save)
        self.im_num_to_save_slider.setMaximum(128)
        self.im_num_to_save_slider.setMinimum(1)
        self.im_num_to_save_txt.setText("save image number %.3f" % self.im_num_to_save)

        # make all_alphas slider
        self.all_alphas = 0
        all_alphas_slider_layout, self.all_alphas_slider, self.all_alphas_txt = self.make_slider_layout(
            self.all_alphas_slider_changed,
            val=self.all_alphas)
        self.all_alphas_txt.setText("all alphas %.3f" % self.all_alphas)

        # make other sliders

        self.num_of_images = 16
        num_of_images_slider_layout, self.num_of_images_slider, self.num_of_images_txt = self.make_slider_layout(self.num_of_images_slider_changed, val=self.num_of_images)
        self.num_of_images_slider.setMinimum(1)
        self.num_of_images_slider.setMaximum(128)
        self.num_of_images_slider.setValue(self.num_of_images)
        self.num_of_images_txt.setText("Num of images = %d" % self.num_of_images)

        # make loop slider
        self.loop_num = 0
        loop_slider_layout, self.loop_slider, self.loop_txt = self.make_slider_layout(self.loop_slider_changed,
                                                                                         val=self.loop_num)
        self.loop_slider.setMinimum(1)
        self.loop_slider.setMaximum(3)
        self.loop_txt.setText("loop num = %.3f" % self.loop_num)

        # make generation Button
        generation_button = QPushButton('Generate')
        generation_button.clicked.connect(self.on_generation_button_click)
        # make activity buttons layout
        activity_buttons_layout = QHBoxLayout()
        activity_buttons_layout.addWidget(generation_button)

        # make save image Button
        save_image_button = QPushButton('Save Image')
        save_image_button.clicked.connect(self.on_save_image_click)

        # save all images
        save_all_button = QPushButton('Save All')
        save_all_button.clicked.connect(self.on_save_all_click)
        # make biro buttons layout
        biro_buttons_layout = QHBoxLayout()
        biro_buttons_layout.addWidget(save_image_button)
        biro_buttons_layout.addWidget(save_all_button)
        # widget layout
        layout = QVBoxLayout()
        layout.addLayout(self.main_layout)
        layout.addLayout(alpha_slider_layout1)
        layout.addLayout(alpha_slider_layout2)
        layout.addLayout(alpha_slider_layout3)
        layout.addLayout(alpha_slider_layout4)
        layout.addLayout(all_alphas_slider_layout)
        layout.addLayout(im_num_to_save_slider_layout)
        layout.addLayout(num_of_images_slider_layout)
        layout.addLayout(loop_slider_layout)
        layout.addLayout(activity_buttons_layout)
        layout.addLayout(biro_buttons_layout)
        self.setLayout(layout)
        # net init
        self.input_tensor = None
        self.output_tensor = None
        self.output_image = None
        self.init_net()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        # show widget
        self.show()

    # change alphas
    def change_alphas(self, alpha):

        self.model.net.alpha1 = alpha
        self.model.net.alpha2 = alpha
        self.model.net.alpha3 = alpha
        self.model.net.alpha4 = alpha

    # all alphas slider changed
    def all_alphas_slider_changed(self):
        self.all_alphas = (self.all_alphas_slider.value() / 100) ** 1

        self.change_alphas(self.all_alphas)

        self.alpha_txt1.setText("alpha1 = %.3f" % self.all_alphas)
        self.alpha_txt2.setText("alpha2 = %.3f" % self.all_alphas)
        self.alpha_txt3.setText("alpha3 = %.3f" % self.all_alphas)
        self.alpha_txt4.setText("alpha4 = %.3f" % self.all_alphas)
        self.all_alphas_txt.setText("alphas = %.3f" % self.all_alphas)

        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    # sliders functions
    def alpha_slider_changed(self):
        self.alpha1 = (self.alpha_slider1.value() / 100) ** 1
        self.alpha2 = (self.alpha_slider2.value() / 100) ** 1
        self.alpha3 = (self.alpha_slider3.value() / 100) ** 1
        self.alpha4 = (self.alpha_slider4.value() / 100) ** 1

        self.model.net.alpha1 = self.alpha1
        self.model.net.alpha2 = self.alpha2
        self.model.net.alpha3 = self.alpha3
        self.model.net.alpha4 = self.alpha4

        self.alpha_txt1.setText("alpha1 = %.3f" % self.alpha1)
        self.alpha_txt2.setText("alpha2 = %.3f" % self.alpha2)
        self.alpha_txt3.setText("alpha3 = %.3f" % self.alpha3)
        self.alpha_txt4.setText("alpha4 = %.3f" % self.alpha4)

        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def im_num_to_save_slider_changed(self):
        self.im_num_to_save = self.im_num_to_save_slider.value()
        self.im_num_to_save_txt.setText("save image number %d" % self.im_num_to_save)

    def num_of_images_slider_changed(self):
        self.num_of_images = self.num_of_images_slider.value()
        self.num_of_images_txt.setText("Num of images = %d" % self.num_of_images)
        self.set_output_image()

    def loop_slider_changed(self):
        self.loop_num = (self.loop_slider.value())
        self.model.net.loop_count = self.loop_num
        self.loop_txt.setText("loop_num = %.3f" % self.loop_num)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    # drop event
    def execute_drop_event(self, file_name):
        pass

    # images functions
    def generate(self):
        return torch.randn((self.opt.batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.model.device)

    def set_output_image(self):
        if self.num_of_images > 32:
            nrow = 16
        else:
            nrow = 8
        image = torchvision.utils.make_grid(self.model.recover_tensor(self.output_tensor[:self.num_of_images, :, :, :]).clamp(min=0.0, max=1), nrow=nrow)
        self.output_image = transforms.ToPILImage()(image.cpu())
        pix_map = self.transformer.pil2pixmap(self.output_image)
        self.generated_image_label.setPixmap(pix_map)

    def init_net(self):
        self.model = BankModel(self.opt)

        path = os.path.join(self.opt.model_save_dir, 'gen_bank_latest.pth')
        self.model.net.load_state_dict(torch.load(path))
        self.model.net.to(self.model.device)
        self.model.net.train()

        path = os.path.join(self.opt.model_save_dir, 'disc_bank_latest.pth')
        self.model.main_disc.load_state_dict(torch.load(path))
        self.model.main_disc.to(self.model.device)
        self.model.main_disc.train()

    def run(self):
        self.output_tensor = self.model.forward(self.input_tensor)

    def on_save_image_click(self):
        gui = QWidget()
        init_text = self.app_name + '_%d_a_%.2f_%.2f_%.2f_%.2f_L_%d' % (self.im_num_to_save,self.alpha1,self.alpha2,self.alpha3,self.alpha4,self.loop_num)
        text, ok = QInputDialog.getText(gui, "save", """ file name """, text=init_text)
        gui.show()
        im_to_save = self.generate_before_after()
        if ok:
            im_to_save.save(os.path.join('gui', 'apps', self.app_name,'saved_output_images', text + '.png'))

    def on_save_interp_click(self):
        gui = QWidget()
        init_text = '0 1 10'
        text, ok = QInputDialog.getText(gui, "save interpolations run", """ write image indexes and number of interpolation images """,
                                        text=init_text)
        gui.show()
        if ok:
            self.make_interp_image(text)
            # output_image.save(os.path.join('gui', 'apps', self.app_name, 'saved_output_images', 'interp_image.png'))

    def make_interp_image(self, text):
        text_list = text.split()
        num_imgs = int(text_list[-1])
        image_num_list = []
        for ind in range(len(text_list) - 1):
            image_num_list.append(int(text_list[ind]))
        im1_idx = image_num_list[0]
        im2_idx = image_num_list[1]

        interval = (self.input_tensor[im2_idx,:,:,:] - self.input_tensor[im1_idx, :, :, :])/(num_imgs-1)
        interp_input_tensor = self.input_tensor.clone()
        for interp in range(num_imgs):
            output_tensor = self.model.forward(interp_input_tensor)
            temp_tensor = torchvision.utils.make_grid(
                self.model.recover_tensor(output_tensor[im1_idx, :, :, :]).clamp(min=0.0, max=1), nrow=1)
            temp_img = transforms.ToPILImage()(temp_tensor.cpu())
            temp_img.save(os.path.join('gui', 'apps', self.app_name, 'saved_output_images', '%dto%d_interp%d_image.png'% (im1_idx,im2_idx,interp)))

            interp_input_tensor[im1_idx, :, :, :] = interval + interp_input_tensor[im1_idx, :, :, :]



    def on_save_alpha_run_click(self):
        gui = QWidget()
        init_text = '0 1 2 3 4 5 6 7 11'
        text, ok = QInputDialog.getText(gui, "save alpha run", """ write image numbers and number of alphas """, text=init_text)
        gui.show()
        if ok:
            self.make_alpha_run_image(text)
            # output_image.save(os.path.join('saved_output_images', '%s_alpha_run.png' % self.app_name))

    def make_alpha_run_image(self, text):
        text_list = text.split()
        num_of_alphas = int(text_list[-1])
        image_num_list = []
        for ind in range(len(text_list) - 1):
            image_num_list.append(int(text_list[ind]))

        directory = os.path.join('gui', 'apps', self.app_name, 'saved_output_images', 'ref%d_interp_%d' % (self.ref_idx, image_num_list[0]))
        if not os.path.exists(directory):
            os.makedirs(directory)
        interval = float(1 / (num_of_alphas - 1))
        for ind in range(num_of_alphas):
            alpha = ind * interval
            if ind == num_of_alphas - 1:
                alpha = 1
            self.change_alphas(alpha)
            output_tensor = self.model.forward(self.input_tensor, alpha=alpha, reference=self.ref_img, img_num =self.im_num_to_save)
            temp_tensor = torchvision.utils.make_grid(self.model.recover_tensor(output_tensor[image_num_list, :, :, :]).clamp(min=0.0, max=1), nrow=1)
            temp_img =transforms.ToPILImage()(temp_tensor.cpu())
            temp_img.save(os.path.join(directory,'alpha_%.2f_image_%d.png' % (alpha, image_num_list[0])))


    def generate_before_after(self):
        # generate image "before"
        self.model.net.loop_count = 1
        output_tensor = self.model.forward(self.input_tensor, alpha=self.alpha1)

        temp_tensor = torchvision.utils.make_grid(
            self.model.recover_tensor(output_tensor[self.im_num_to_save, :, :, :]).clamp(min=0.0, max=1), nrow=1)
        image_tensor = temp_tensor

        # generate image "after"
        self.model.net.loop_count = self.loop_num
        output_tensor = self.model.forward(self.input_tensor, alpha=self.alpha1)
        temp_tensor = torchvision.utils.make_grid(
            self.model.recover_tensor(output_tensor[self.im_num_to_save, :, :, :]).clamp(min=0.0, max=1), nrow=1)
        image_tensor = torch.cat([image_tensor, temp_tensor], 2)

        return transforms.ToPILImage()(image_tensor.cpu())

    def on_save_all_click(self):
        init_text = self.app_name + '_a_%.2f_%.2f_%.2f_%.2f_L_%d' % (self.model.net.alpha1, self.model.net.alpha2, self.model.net.alpha3, self.model.net.alpha4, self.loop_num)
        self.output_image.save(os.path.join('gui', 'apps', self.app_name,'saved_output_images', init_text+'.png'))

    def on_generation_button_click(self):
        self.generation_noise = self.generate()
        self.input_tensor = self.generation_noise
        self.run()
        self.set_output_image()

    def generate_for_eval(self):
        samples_needed = 50000
        iters = samples_needed // self.opt.batch_size
        dir_name = 'a%.2f_%.2f_%.2f_%.2f_L_%d' % (self.alpha1, self.alpha2, self.alpha3, self.alpha4, self.loop_num)
        dir_path = os.path.join('gui', 'apps', self.app_name, dir_name)
        utils.make_dirs(dir_path)
        im_count = 0

        for iter in range(iters):
            self.generation_noise = self.generate()
            self.input_tensor = self.generation_noise
            self.run() # result exists in self.output_tensor

            tensor = self.model.recover_tensor(self.output_tensor).clamp(min=0.0, max=1)

            for idx in range(self.opt.batch_size):
                img = transforms.ToPILImage()(tensor[idx, :, :, :].cpu())
                img.save(os.path.join(dir_path, '%d.png' % im_count))
                im_count = im_count + 1







