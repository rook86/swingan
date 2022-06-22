import argparse
import os
import numpy as np
import math
import itertools
import sys

from torchvision.utils import save_image, make_grid

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.autograd import Variable

#from generator import *
from discriminator import *
from feature_extractor import *
from losses import *
from dataloader import *
from metrics import *
from utils import *
from swinir import * 
from option import get_option
from metrics import compute_psnr, compute_ssim

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("test_images", exist_ok=True)

opt = get_option()

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

upscale = 4
window_size = 8
height = (256 // upscale // window_size + 1) * window_size
width = (256 // upscale // window_size + 1) * window_size

# Initialize generator
generator = SwinIR(upscale=4, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to('cuda')

device = torch.device("cuda")



# Load pretrained models
model_dict = "saved_models/generator_0.pth"
generator.load_state_dict(torch.load(model_dict))
if cuda:
    generator = generator.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../sr_datasets/set5", hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Testing
# ----------

total_psnr = 0
total_ssim = 0
for i, imgs in enumerate(dataloader):

    # Configure model input
    imgs_lr = Variable(imgs["lr"].type(Tensor))
    imgs_hr = Variable(imgs["hr"].type(Tensor))

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

    # Generate a high resolution image from low resolution input
    gen_hrs = []
    for j in range(len(chunks_lr)):
        gen_hrs.append(generator(chunks_lr[j]))

    # --------------
    #  Log Progress
    # --------------

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
    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
    gen_hr_cp = tensor2np(gen_hr.detach()[0])
    imgs_hr_cp = tensor2np(imgs_hr.detach()[0])
    psnr = compute_psnr(gen_hr_cp, imgs_hr_cp)
    ssim = compute_ssim(gen_hr_cp, imgs_hr_cp)
    print("[PSNR %f] [SSIM %f]" % (psnr, ssim))
    total_psnr += psnr
    total_ssim += ssim
    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
    save_image(img_grid, "test_images/%d.png" % i)

print("[AVERAGE_PSNR %f] [AVERAGE_SSIM %f]" % (total_psnr/len(dataloader), total_ssim/len(dataloader)))
