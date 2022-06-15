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

from generator import *
from discriminator import *
from feature_extractor import *
from losses import *
from dataloader import *
from option import get_option
import augments

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

cuda = torch.cuda.is_available()

opt = get_option()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = Generator(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=10, num_grow_ch=32)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))

# Losses
perceptual_loss = PerceptualLoss()
gan_loss = GANLoss()
criterion_MSE = torch.nn.MSELoss()
attention_loss = AttentionLoss()

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
    criterion_attention = attention_loss.cuda()

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

        imgs_hr, imgs_lr, mask, aug = augments.apply_augment(
                imgs_hr, imgs_lr,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )
        
        trans = transforms.Resize((512//4, 512//4), Image.BICUBIC)
        imgs_lr = trans(imgs_lr)

        chunk_dim = 2
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

        

        # Generate a high resolution image from low resolution input
        total_loss_G = 0
        gen_hrs = []
        for j in range(len(chunks_lr)):
            optimizer_G.zero_grad()
            gen_hrs.append(generator(chunks_lr[j]))
            fake_pred = discriminator(generator(chunks_lr[j]))
            loss_GAN = criterion_gan(fake_pred, True, is_disc=False)
            loss_content = criterion_perceptual(generator(chunks_lr[j]), chunks_hr[j])
            loss_pixel = criterion_MSE(generator(chunks_lr[j]), chunks_hr[j])
            #loss_attention = criterion_attention(generator(chunks_lr[j]), chunks_hr[j])
            # Total loss 
            loss_G = loss_GAN + loss_pixel + loss_content #+ loss_attention
            total_loss_G += loss_G
            loss_G.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        

        # Loss of real and fake images
        total_loss_D = 0
        for j in range(len(chunks_lr)):
            optimizer_D.zero_grad()
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
            """
            h1 = torch.cat((gen_hrs[0],gen_hrs[1],gen_hrs[2],gen_hrs[3]),3)
            h2 = torch.cat((gen_hrs[4],gen_hrs[5],gen_hrs[6],gen_hrs[7]),3)
            h3 = torch.cat((gen_hrs[8],gen_hrs[9],gen_hrs[10],gen_hrs[11]),3)
            h4 = torch.cat((gen_hrs[12],gen_hrs[13],gen_hrs[14],gen_hrs[15]),3)
            h12 = torch.cat((h1,h2),2)
            h34 = torch.cat((h3,h4),2)
            gen_hr = torch.cat((h12,h34),2)
            """
            up = torch.cat((gen_hrs[0],gen_hrs[1]),3)
            down = torch.cat((gen_hrs[2],gen_hrs[3]),3)
            gen_hr = torch.cat((up,down),2)
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


