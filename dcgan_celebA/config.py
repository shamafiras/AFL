import argparse
import host_configurations.paths as paths
import os


def get_configurations():
    # set configurations here
    experiment_name = 'pretrained'  # write here the name of the experiment
    data_set = 'celebA'
    roots_dict = paths.get_host_paths(data_set)
    main_dir_name = os.path.join('celeb', experiment_name)
    main_epochs = 20
    bank_epochs = 20
    batch_size = 128
    z_size = 100
    gen_learning_rate_main = 0.0002
    gen_learning_rate_bank = 0.0002
    disc_learning_rate_main = 0.0002
    disc_learning_rate_bank = 0.0002
    image_size = 64
    bank_disc_same_as_main_disc = False
    crop_type = '108'

    discriminator_main_attr = 'All'
    discriminator_bank_attr = 'All'
    discriminator_main_attr_is = True
    discriminator_bank_attr_is = False

    training_scheme = 'bank'

    cuda = True
    eval_noise_batch_size = 128

    eval_iter = 20
    intermediate_images_iter = 20000
    save_image_iter = 200

    data_set_path = roots_dict['data_set']
    attr_path = roots_dict['attr']

    model_save_dir = os.path.join(main_dir_name, 'model_dir')
    images_save_dir = os.path.join(main_dir_name, 'images')
    pre_trained_model = os.path.join(model_save_dir, 'original_main_latest.pth')
    # pre_disc_trained_model = 'none'
    pre_disc_trained_model = os.path.join(model_save_dir, 'disc_main_latest.pth')


    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default=data_set)
    parser.add_argument('--discriminator_main_attr', default=discriminator_main_attr)
    parser.add_argument('--discriminator_bank_attr', default=discriminator_bank_attr)
    parser.add_argument('--discriminator_main_attr_is', default=discriminator_main_attr_is)
    parser.add_argument('--discriminator_bank_attr_is', default=discriminator_bank_attr_is)
    parser.add_argument('--crop_type', default=crop_type)
    parser.add_argument('--bank_disc_same_as_main_disc', default=bank_disc_same_as_main_disc, type=bool)
    parser.add_argument('--main_epochs', default=main_epochs, type=int)
    parser.add_argument('--bank_epochs', default=bank_epochs, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--image_size', default=image_size, type=int)
    parser.add_argument('--eval_noise_batch_size', default=eval_noise_batch_size, type=int)
    parser.add_argument('--z_size', default=z_size, type=int)
    parser.add_argument('--gen_learning_rate_main', default=gen_learning_rate_main, type=float)
    parser.add_argument('--gen_learning_rate_bank', default=gen_learning_rate_bank, type=float)
    parser.add_argument('--disc_learning_rate_main', default=disc_learning_rate_main, type=float)
    parser.add_argument('--disc_learning_rate_bank', default=disc_learning_rate_bank, type=float)
    parser.add_argument('--eval_iter', default=eval_iter, type=int)
    parser.add_argument('--intermediate_images_iter', default=intermediate_images_iter, type=int)
    parser.add_argument('--save_image_iter', default=save_image_iter, type=int)
    parser.add_argument('--data_set_path', default=data_set_path)
    parser.add_argument('--attr_path', default=attr_path)
    parser.add_argument('--model_save_dir', default=model_save_dir)
    parser.add_argument('--images_save_dir', default=images_save_dir)
    parser.add_argument('--main_dir_name', default=main_dir_name)
    parser.add_argument('--pre_trained_model', default=pre_trained_model)
    parser.add_argument('--pre_disc_trained_model', default=pre_disc_trained_model)
    parser.add_argument('--cuda', default=cuda, type=bool)
    parser.add_argument('--training_scheme', default=training_scheme)
    parser.add_argument('--loop_count', default=1, type=int)
    parser.add_argument('--a1', type=float, default=1)
    parser.add_argument('--a2', type=float, default=1)
    parser.add_argument('--a3', type=float, default=1)
    parser.add_argument('--a4', type=float, default=1)
    parser.add_argument('--app_name', type=str, default='')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    opt = parser.parse_args()
    return opt
