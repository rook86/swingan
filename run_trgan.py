import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary

from esrt import ESRT
from discriminator import *
from feature_extractor import *
from losses import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

def set_gpu(model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    # DataParallel will divide and allocate batch_size to all available GPUs
    torch.cuda.set_device(2)
    model = torch.nn.DistributedDataParallel(model, device_ids=[2,3]).cuda(2)

    cudnn.benchmark = True

    return model

os.makedirs("train_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=512, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = ESRT()#architecture.IMDN(upscale=args.scale)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))

# Losses
perceptual_loss = PerceptualLoss()
gan_loss = GANLoss()
criterion_MSE = torch.nn.MSELoss()

device = torch.device("cuda")

if cuda:
    #generator = set_gpu(generator)
    #discriminator = set_gpu(discriminator)
    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)
    generator.to(device)
    discriminator.to(device)
    criterion_MSE = criterion_MSE.cuda()
    criterion_perceptual = perceptual_loss.cuda()
    criterion_gan = gan_loss.cuda()

# Load pretrained models
# generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
# discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../sr_datasets/DIV2K_train_HR", hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        chunk_dim = 4
        a_x_split = torch.chunk(imgs_lr, chunk_dim, dim=2)

        chunks_lr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_lr.append(c_)
        
        a_x_split = torch.chunk(imgs_hr, chunk_dim, dim=2)

        chunks_hr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_hr.append(c_)

        # Adversarial ground truths
        # valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        total_loss_G = 0
        gen_hrs = []
        for j in range(len(chunks_lr)):
            gen_hrs.append(generator(chunks_lr[j]))
            fake_pred = discriminator(generator(chunks_lr[j]))
            loss_GAN = criterion_gan(fake_pred, True, is_disc=False)
            loss_content = criterion_perceptual(generator(chunks_lr[j]), chunks_hr[j])
            loss_pixel = criterion_MSE(generator(chunks_lr[j]), chunks_hr[j])
            # Total loss
            loss_G = loss_GAN + loss_pixel + loss_content
            total_loss_G += loss_G
            loss_G.backward()
            optimizer_G.step()
        gen_hr = generator(imgs_lr)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        total_loss_D = 0
        for j in range(len(chunks_lr)):
            real_pred = discriminator(chunks_hr[j])
            fake_pred = discriminator(generator(chunks_lr[j]))
            loss_real = criterion_gan(real_pred, True, is_disc=True)
            loss_fake = criterion_gan(fake_pred, False, is_disc=True)
            # Total loss
            loss_D = loss_real + loss_fake
            total_loss_D += loss_D
            loss_D.backward()
            optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                total_loss_D/16,
                total_loss_G/16
            )
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            h1 = torch.cat((gen_hrs[0],gen_hrs[1],gen_hrs[2],gen_hrs[3]),3)
            h2 = torch.cat((gen_hrs[4],gen_hrs[5],gen_hrs[6],gen_hrs[7]),3)
            h3 = torch.cat((gen_hrs[8],gen_hrs[9],gen_hrs[10],gen_hrs[11]),3)
            h4 = torch.cat((gen_hrs[12],gen_hrs[13],gen_hrs[14],gen_hrs[15]),3)
            h12 = torch.cat((h1,h2),2)
            h34 = torch.cat((h3,h4),2)
            gen_hr = torch.cat((h12,h34),2)
            PSNR = 10 * math.log(255*255/criterion_MSE(10*imgs_hr, 10*gen_hr),10)
            print("[PSNR %f]" % (PSNR))
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
            save_image(img_grid, "train_images/%d.png" % batches_done)

    torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)


