import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from feature_extractor import VGGFeatureExtractor
import common

class PerceptualLoss(nn.Module):

    def __init__(self,
                layer_weights={'conv1_2': 0.1, 
                                'conv2_2': 0.1,
                                'conv3_4': 1,
                                'conv4_4': 1,
                                'conv5_4': 1},
                vgg_type='vgg19',
                use_input_norm=True,
                range_norm=False,
                perceptual_weight=1.0,
                style_weight=0.,
                criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        return percep_loss

    def _gram_mat(self, x):
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):

    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_label(self, input, target_is_real):
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        #return self.pool(x) - x
        return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)


class AttentionLoss(nn.Module):
    def __init__(self, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = 288
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.pool = Pooling()
        #self.reduce = common.default_conv(3, 3, 5)
        self.qkv = nn.Linear(dim//2, dim//2 * 3, bias=qkv_bias)

        criterion_mse = torch.nn.MSELoss()
    
    def split(self, x):
        #x = self.reduce(x)
        x = torch.squeeze(x, 3)
        x = self.pool(x)
        x = self.pool(x)
        print(x.shape)
        D,B,N,C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        return q, k, v

    def forward(self, g, hr):
        
        q_g, k_g, v_g = self.split(g)
        q_hr, k_hr, v_hr = self.split(hr)
        loss =  criterion_MSE(q_g, q_hr) + criterion_MSE(k_g, k_hr) + criterion_MSE(k_g, k_hr)
        return loss
