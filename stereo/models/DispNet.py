#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
#import torch.nn.functional as F
#from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import BaseModule as BM
import torch.nn.functional as F
import traceback
import logging
from torchvision.utils import make_grid
logger = logging.getLogger(__name__)


visualize_disps = False # True # 

visualize_attention = False  # True #
visualize_refine = False  # True #

ActiveFun = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True) # 
NormFun2d = nn.BatchNorm2d # nn.InstanceNorm2d # 
def conv1x1(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1,
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
   # kargs['padding'] = kargs['dilation']
    return nn.Conv2d(in_channel, out_channel, 1, **kargs)


def conv1x1_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
        conv1x1(in_channel, out_channel, **kargs),
        NormFun2d(out_channel), ActiveFun, )

def conv5x5(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1,
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    kargs['padding'] = 2
    return nn.Conv2d(in_channel, out_channel, 5, **kargs)


def conv5x5_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
        conv5x5(in_channel, out_channel, **kargs),
        NormFun2d(out_channel), ActiveFun, )

def conv7x7(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1,
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    kargs['padding'] = 3
    return nn.Conv2d(in_channel, out_channel, 7, **kargs)


def conv7x7_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
        conv7x7(in_channel, out_channel, **kargs),
        NormFun2d(out_channel), ActiveFun, )
def conv9x9(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1,
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    kargs['padding'] = 4
    return nn.Conv2d(in_channel, out_channel, 9, **kargs)


def conv9x9_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
        conv9x9(in_channel, out_channel, **kargs),
        NormFun2d(out_channel), ActiveFun, )
def conv3x3(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1,
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    kargs['padding'] = kargs['dilation']
    return nn.Conv2d(in_channel, out_channel, 3, **kargs)


def conv3x3_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
        conv3x3(in_channel, out_channel, **kargs),
        NormFun2d(out_channel), ActiveFun, )

def padConv2d(in_channel, out_channel, kernel_size, **kargs):
    """
    Conv2d with padding
    >>> module = padConv2d(1, 1, 5, stride=2, groups=1, bias=True) # dilation=1, 
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    pad = kernel_size//2
    if(kargs['dilation'] > 1): pad *= kargs['dilation']
    kargs['padding'] = pad
    return nn.Conv2d(in_channel, out_channel, kernel_size, **kargs)


def padConv2d_bn(in_channel, out_channel, kernel_size, **kargs):
    """
    Conv2d with padding, BatchNorm and ActiveFun
    >>> module = padConv2d_bn(1, 1, 5, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
                padConv2d(in_channel, out_channel, kernel_size, **kargs), 
                NormFun2d(out_channel), ActiveFun, )


def deconv2d(in_channel, out_channel, kernel_size, stride=2):

    padding = (kernel_size - 1)//2
    output_padding = stride - (kernel_size - 2*padding)
    return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, bias=False, 
                                padding=padding, output_padding=output_padding), 
            ActiveFun, 
            )


def corration1d(fL, fR, shift, stride=1):
    """
    corration of left feature and shift right feature
    
    corration1d(tensor, shift=1, dim=-1) --> tensor of 4D corration

    Args:

        fL: 4D left feature
        fR: 4D right feature
        shift: count of shift right feature
        stride: stride of shift right feature


    Examples:
    
        >>> x = torch.rand(1, 3, 32, 32)
        >>> y = corration1d(x, x, shift=20, stride=1)
        >>> list(y.shape)
        [1, 20, 32, 32]
    """
    
    bn, c, h, w = fL.shape
    corrmap = torch.zeros(bn, shift, h, w).type_as(fL.data)
    for i in range(0, shift):
        idx = i*stride
        corrmap[:, i, :, idx:] = (fL[..., idx:]*fR[..., :w-idx]).mean(dim=1)
    
    return corrmap
    
class Kd(nn.Module):
    def __init__(self,planes,di,stride):
        super(Kd,self).__init__()
        self.di=di
        self.planes=planes
        self.stride=stride
        self.conv=conv3x3_bn(planes,planes)
        self.d=conv1x1_bn(planes, planes, stride=stride)
    #    self.out=conv1x1_bn(planes,planes,stride=stride)
        if di==2:
            self.kd=conv3x3_bn(planes,planes,dilation=di)
            self.cat=conv3x3_bn(2*planes,planes,stride=stride)
        if di==3:
            self.kd1=conv3x3_bn(planes,planes,dilation=di-1)
            self.kd2=conv3x3_bn(planes,planes,dilation=di)
            self.cat=conv3x3_bn(3*planes,planes,stride=stride)
        if di==4:
            self.kd1=conv3x3_bn(planes,planes,dilation=di-2)
            self.kd2=conv3x3_bn(planes,planes,dilation=di-1)
            self.kd3=conv3x3_bn(planes,planes,dilation=di)
            self.cat=conv3x3_bn(4*planes,planes,stride=stride)
        if di==5:
            self.kd1=conv3x3_bn(planes,planes,dilation=di-3)
            self.kd2=conv3x3_bn(planes,planes,dilation=di-2)
            self.kd3=conv3x3_bn(planes,planes,dilation=di-1)
            self.kd4=conv3x3_bn(planes,planes,dilation=di)
            self.cat=conv3x3_bn(5*planes,planes,stride=stride)
    def forward(self,x):
        if self.di==2:
            kd=self.kd(x)
        if self.di==3:
            kd1=self.kd1(x)
            kd2=self.kd2(x)
            kd=torch.cat([kd1,kd2],dim=1)
        if self.di==4:
            kd1=self.kd1(x)
            kd2=self.kd2(x)
            kd3=self.kd3(x)
            kd=torch.cat([kd1,kd2,kd3],dim=1)
        if self.di==5:
            kd1=self.kd1(x)
            kd2=self.kd2(x)
            kd3=self.kd3(x)
            kd4=self.kd4(x)
            kd=torch.cat([kd1,kd2,kd3,kd4],dim=1)
        conv=self.conv(x)
        d=self.d(x)
        out=torch.cat([kd,conv],dim=1)
        out=self.cat(out)
        out=out+d
        #out=self.out(out)
        return out
class AttentionP(nn.Module):
    
    def __init__(self,planes):
        super(AttentionP,self).__init__()
        self.planes=planes
        
        ##上
        self.gp=nn.AvgPool2d(2)
        self.s=conv1x1_bn(planes, planes)
        #中
        self.z=conv1x1_bn(planes, planes)
        #下

        self.x11=Kd(planes,3,2)

        self.x12=Kd(planes,3,1)

        self.x21=Kd(planes,2,2)

        self.x22=Kd(planes,2,1)
        self.x31=conv3x3_bn(planes, planes, stride=2)
        self.x32=conv3x3_bn(planes, planes, stride=1)
    def forward(self,x):
        #上
        up=self.gp(x)
        up=self.s(up)
        up=BM.upsample_as_bilinear(up,x)
        
        #中
        mid=self.z(x)
        
        #下
        d1=self.x11(x)
        d2=self.x21(d1)
        d3=self.x31(d2)
        d1=self.x12(d1)
        d2=self.x22(d2)
        d3=self.x32(d3)
        d3=BM.upsample_as_bilinear(d3,d2)
        d2=d3+d2
        d2=BM.upsample_as_bilinear(d2,d1)
        d1=d2+d1
        d1=BM.upsample_as_bilinear(d1,x)
        
        #合
        out=d1*mid
        out=out+up
        return out
class DispNetT(BM.BaseModule):
    def __init__(self,args):
        super(DispNetT,self).__init__()
        assert self._lossfun_is_valid(args.loss_name), \
            'invalid lossfun [ model: %s, lossfun: %s]'%(args.arch, args.loss_name)
        self.maxdisp = args.maxdisp
        self.first=nn.Sequential(conv3x3_bn(3, 32),
                                 conv3x3_bn(32, 32,stride=2))
        self.conv1a=Kd(32,2)
        self.conv2a=Kd(32,3)
        self.conv3a=Kd(32,4)
        
        #
        self.fpa=AttentionP(32)
        
        self.conv1b=conv3x3_bn(32, 32)
        self.conv2b=conv3x3_bn(32, 32)
        self.conv3b=conv3x3_bn(32, 32)
        self.conv4=conv3x3_bn(32, 32)
        
        
    def forward(self,imL,imR):
        bn=imL.size(0)
        x=torch.cat([imL,imR],dim=0)
        f=self.first(x)
        f1=self.conv1a(x)
        f2=self.conv2a(x)
        f3=self.conv3a(x)
        fp=self.fpa(x)
        disp=BM.upsample_as_bilinear(fp, f2)
        disp=self.conv1b(disp)
        disp=BM.upsample_as_bilinear(disp, f1)
        disp=self.conv2b(disp)
        disp=BM.upsample_as_bilinear(disp, f)
        disp=self.conv3b(disp)
        disp=BM.upsample_as_bilinear(disp, imL)
        disp=self.conv4(disp)

class AttentionSPP(nn.Module):
    """Spatial pyramid pooling
    >>> feature = torch.rand(2, 16, 5, 5)
    >>> msf = AttentionSPP(16, nLevel=4, kernel_size_first=2)
    >>> out = msf(feature)
    >>> list(out.shape)
    [2, 16, 5, 5]
    """

    def __init__(self, planes, nLevel=4, kernel_size_first=2, attention=True):
        super(AttentionSPP, self).__init__()

        self.planes = planes
        self.nLevel = nLevel
        self.kernel_size_first = kernel_size_first
        self.attention = attention

        ks = [kernel_size_first * (2 ** i) for i in range(nLevel)]
        self.avg_pools = nn.ModuleList([nn.AvgPool2d(tks) for tks in ks])  # , ceil_mode=True
        self.branchs = nn.ModuleList([conv3x3_bn(planes, planes) for i in range(nLevel + 1)])

        self.lastconv = conv3x3_bn(planes * (nLevel + 1), planes)

    def forward(self, x):

        h, w = x.shape[-2:]
        kargs_up = {'size': (h, w), 'mode': 'bilinear', 'align_corners': True}
        upsamples = [lambda x: x] + [lambda x: F.interpolate(x, **kargs_up)] * self.nLevel
        avg_pools = [lambda x: x] + list(self.avg_pools)
        fun_branch = lambda upsample, branch, avg_pool: upsample(branch(avg_pool(x)))
        branchs = list(map(fun_branch, upsamples, self.branchs, avg_pools))

        if (not self.attention):
            return self.lastconv(torch.cat(branchs, dim=1))

        weights = torch.cat([branchs[i][:, :1] for i in range(len(branchs))], dim=1)
        weights = torch.softmax(weights, dim=1)
        branchs = [branchs[i][:, 1:] * weights[:, i:i + 1] for i in range(len(branchs))]
        output = self.lastconv(torch.cat([weights] + branchs, dim=1))

        # visualize attention
        if (visualize_attention):
            imgs = weights[:1]
            h, w = imgs.shape[-2:]
            imgs = imgs.view(-1, 1, h, w)
            pad = max(1, max(h, w) // 100)
            imgs = make_grid(imgs, nrow=1, padding=pad, normalize=False)
            timg = imgs.data.cpu().numpy()[0]
            # path_save = 'z_ASPP_%d_%d.png' % (h//10, w//10)
            # plt.imsave(path_save, timg)
            plt.subplot(111);
            plt.imshow(timg)
            plt.show()

        return output

class DispNet(BM.BaseModule):

    def __init__(self, args):
        super(DispNet, self).__init__()

        assert self._lossfun_is_valid(args.loss_name), \
            'invalid lossfun [ model: %s, lossfun: %s]'%(args.arch, args.loss_name)
        
        self.maxdisp = args.maxdisp
        self.scales = [ 3, 2, 1, 0, ]

        # Feature extration

        self.conv1a=Kd(128,2,2)
        self.conv2a=Kd(128,2,2)
        self.conv3a=Kd(128,2,2)

        self.gd= padConv2d_bn(128, 128, kernel_size=3, stride=1)
        self.fpa=AttentionP(128)
        
        
        self.conv1b=conv3x3_bn(256, 64)
        self.conv2b=conv3x3_bn(192, 64)
        self.conv3b=conv3x3_bn(128, 32)
        self.conv4=conv3x3_bn(64, 32)
        
        self.d1=padConv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.d2=padConv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.d3=padConv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.d4=padConv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        
        self.q1=conv3x3_bn(6, 32)
        self.q2=Kd(32,2,1)
        self.q3=Kd(32,2,1)
        self.q4=Kd(32,2,1)
        # self.q2=conv5x5_bn(32,32)
        # self.q3=conv7x7_bn(32,32)
        self.aspp=AttentionSPP(32, nLevel=4, kernel_size_first=4, attention=True)
        self.h1=Kd(32,2,1)
        # self.h1=conv5x5_bn(32,32)
        self.h2=conv3x3_bn(32, 1)
        
        #
        self.re1=conv3x3_bn(1, 1,stride=2)
        self.re2=conv3x3_bn(1, 1,stride=2)
        self.re3=conv3x3_bn(1, 1,stride=2)

    def _lossfun_is_valid(self, loss_name):

        loss_name = loss_name.lower()
        invalid1 = (loss_name.startswith('sv')) and ('ce' in loss_name)
        invalid2 = (loss_name.startswith('lusv')) and ('ec' in loss_name)
        invalid = invalid1 or invalid2
        return (not invalid)


    def _pr_zoom_k(self, k=0.1):
        for m in [self.d1,self.d2,self.d3,self.d4]:
            m.weight.data = m.weight.data*k


    def _modules_weight_decay(self):
        return [
                self.conv1a, self.conv2a, self.conv3a
                ]


    def _modules_conv(self):
        return [
            self.conv1b,self.conv2b,self.conv3b,
            self.conv4,self.gd,self.fpa,self.d1,self.d2,self.d3,self.d4,self.aspp,
            self.q1,self.q2,self.q3,self.q4,self.h1,self.h2,self.re1,self.re2,self.re3
                ]


    def _disp_estimator(self, imL,imR, fconv1, fconv2, fconv3):

        # replace None for returns. 
        # returns contain of None will cause exception with using nn.nn.DataParallel
        invalid_value = torch.zeros(1).type_as(imL) 

        # feature extration
        
        bn=imL.size(0)
       # x=torch.cat([imL,imR],dim=0)
       # f=self.first(x)
      #  f1=self.spp(fconv3)
        f1=self.conv1a(fconv3)
        f2=self.conv2a(f1)
        f3=self.conv3a(f2)
     #   f3=self.conv4a(f3)
        
     #   f3=self.conv4a(f3)
        fp=self.fpa(f3)
        conv4=BM.upsample_as_bilinear(fp, f2)
        conv4=torch.cat([conv4,f2],dim=1)
        conv4=self.conv1b(conv4)
        disp4=self.d4(conv4)
        conv3=BM.upsample_as_bilinear(conv4, f1)
        conv3=torch.cat([conv3,f1],dim=1)
        conv3=self.conv2b(conv3)
        disp3=self.d3(conv3)
        conv2=BM.upsample_as_bilinear(conv3, fconv2)
        conv2=torch.cat([conv2,fconv2],dim=1)
        conv2=self.conv3b(conv2)
        disp2=self.d2(conv2)
        conv1=BM.upsample_as_bilinear(conv2, fconv1)
        conv1=torch.cat([conv1,fconv1],dim=1)
        conv1=self.conv4(conv1)
        disp1=self.d1(conv1)
        
        
        disp = disp1.detach()
        factor = 2.0 / (disp.size(-1) - 1)
        fL_wrap = BM.imwrap(imR.detach(), disp * factor)
        fx = torch.cat([imL, fL_wrap], dim=1)
        fx=self.q1(fx)
        fx=self.q2(fx)
        fx=self.q3(fx)
        fx=self.q4(fx)
        dr=self.aspp(fx)
        dr=self.h1(dr)
        dr=self.h2(dr)
        disp1=dr+disp1
        dr1=self.re1(dr)
        disp2=disp2+dr1
        dr2=self.re2(dr1)
        disp3=disp3+dr2
        dr3=self.re3(dr2)
        disp4=disp4+dr3

        # visualize_disps
        if(visualize_disps):

            disps = [disp4, disp3, disp2, disp1]
            plt.subplot(111)
            to_numpy = lambda tensor: tensor.cpu().data.numpy()
            col = 2
            row = (len(disps)+col+1)//col

            plt.subplot(row, col, 1); plt.imshow(to_numpy(BM.normalize(imL[0])).transpose(1, 2, 0))
            
            disps.reverse()
            for idx, tdisp in enumerate(disps):
                timg = to_numpy(tdisp[0, 0])
                plt.imsave('z_disp%d.png'%idx, timg)
                plt.subplot(row, col, idx+3); plt.imshow(timg);
            plt.show()
        
        # return
        logger.debug('training: %s \n' % str(self.training))
        if(self.training):
            loss_ex = invalid_value
            return loss_ex, [ disp4/8, disp3/4, disp2/2,disp1]
        else:
            return disp1.clamp(0)


class DispNetS(DispNet):

    def __init__(self, args):
        super(DispNetS, self).__init__(args)

        # 卷积层
        self.conv1 = padConv2d_bn(  6,  64, kernel_size=7, stride=2)
        self.conv2 = padConv2d_bn( 64, 128, kernel_size=5, stride=2)
        self.conv3 = padConv2d_bn(128, 256, kernel_size=5, stride=2)

        # initialize weight
        self.modules_init_()
      #  self._pr_zoom_k(0.1)

    
    def get_parameters(self, lr=1e-3,  weight_decay=0):

        modules_new = [self.conv1, self.conv2, self.conv3, ]
        modules_weight_decay = modules_new + self._modules_weight_decay()
        modules_conv = self._modules_conv()
        return self._get_parameters_group(modules_weight_decay, modules_conv, lr,  weight_decay)


    def forward(self, imL, imR):

        # feature extration
        x = torch.cat([imL, imR], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        # disparity estimator
        return self._disp_estimator(imL, conv1, conv2, conv3)


class ACM(DispNet):

    def __init__(self, args):
        super(ACM, self).__init__(args)

        # maximum of disparity
        self.shift = 40 # 1 + maxdisp//4

        # 卷积层
        self.conv1 = conv3x3_bn(3,32)
        self.convt1 = Kd(32,2,1)
        self.convt2 = Kd(32,3,1)

        self.conv2 = conv3x3_bn(32,64,stride=2)
        self.redir = conv3x3_bn(64,64)
        self.conv3 = conv3x3_bn(64+self.shift,128)

        # initialize weight
        self.modules_init_()
        self._pr_zoom_k(0.1)

    def get_parameters(self, lr=1e-3,  weight_decay=0):

        modules_new = [self.conv1,self.convt1,self.convt2,self.conv2, self.redir, self.conv3, ]
        modules_weight_decay = modules_new + self._modules_weight_decay()
        modules_conv = self._modules_conv()
        return self._get_parameters_group(modules_weight_decay, modules_conv, lr,  weight_decay)


    def forward(self, imL, imR):

        # feature extration
        bn = imL.size(0)
        x = torch.cat([imL, imR], dim=0)
        conv1 = self.conv1(x)
        conv1=self.convt1(conv1)
        conv1=self.convt2(conv1)
        conv2 = self.conv2(conv1)
        redir = self.redir(conv2[:bn])
        corrmap = corration1d(conv2[:bn], conv2[bn:], self.shift, 1)
        conv3 = self.conv3(torch.cat([redir, corrmap], dim=1))
        
        # disparity estimator
        return self._disp_estimator(imL,imR, conv1[:bn], conv2[:bn], conv3)


def get_model_by_name(args):
    
    tmp = args.arch.split('_')
    name_class = tmp[0]
    try:
        return eval(name_class)(args)
    except:
        raise Exception(traceback.format_exc())


def get_settings():

    import argparse
    parser = argparse.ArgumentParser(description='Deep Stereo Matching by pytorch')

    # arguments of model
    parser.add_argument('--arch', default='ACM',
                        help='select arch of model')
    parser.add_argument('--maxdisp', type=int ,default=192,
                        help='maxium disparity')

    # arguments of lossfun
    parser.add_argument('--loss_name', default='SV-SL1',
                        help='name of lossfun [SV-(SL1/CE/SL1+CE), USV-(common/depthmono/SsSMnet/AS1C/AS2C)-mask]')
    parser.add_argument('--flag_FC', action='store_true', default=False,
                        help='enables feature consistency')

    # parser arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    args = get_settings()
    list_name = ['DispNetS', 'ACM' ]

    for name in list_name:
        args.arch = name
        model = get_model_by_name(args)
        logger.info('%s passed!\n ' % model.name)

