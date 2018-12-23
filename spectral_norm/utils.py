import torch
from torch import autograd
import tflib as lib
import tflib.save_images
import numpy as np
import os
from torch import is_tensor
from torch.autograd import Variable
import torchvision as tv

def get_inception_score(G, output_dir, fname):
    all_samples = []
    for i in range(500):
        samples_100 = torch.randn(100, 128, 1, 1)
        samples_100 = samples_100.cuda(0)
        samples_100 = autograd.Variable(samples_100)
        with torch.no_grad():
            all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    if True:
        file = open(output_dir+fname, 'w')
        all_samples.tofile(file)
        file.close()
    else:
        return lib.inception_score.get_inception_score(list(all_samples))

def inception_score_from_file(output_dir,fname):
    all_samples = np.fromfile(output_dir+fname, dtype=np.dtype((np.int32, (32, 32, 3))))
    return lib.inception_score.get_inception_score(list(all_samples))

# For generating samples
def generate_image(frame, netG, output_dir):
    fixed_noise_128 = torch.randn(128, 128,1,1)
    fixed_noise_128 = fixed_noise_128.cuda(0)
    noisev = autograd.Variable(fixed_noise_128)
    with torch.no_grad():
        samples = netG(noisev)
    tv.utils.save_image(samples.data, '%s/%s.png' % (output_dir, frame), normalize=True)

def save_images(samples,output_dir, frame ):
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    if not os.path.exists(output_dir + '/samples'):
        os.mkdir(output_dir + '/samples')
    lib.save_images.save_images(samples, output_dir + '/samples/samples_{}.jpg'.format(frame))

def save_intermediate_res(samples, output_dir, frame):
    i = 0
    for inter in samples:
        save_images(samples=inter, output_dir=output_dir, frame=frame+'%d' % i)
        i = i+1