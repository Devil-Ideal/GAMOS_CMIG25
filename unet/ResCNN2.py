import os
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class Conv3d_wd(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3),out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads.cuda()], dim=1))
    return out
def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)
    return out
def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'relu':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)
    return out
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False,add_time = False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.downsample = downsample
        self.stride = stride
        self.temb_proj = torch.nn.Linear(512,planes)
        self.add_time = add_time
    def forward(self, input):
        x, temb = input[0], input[1]
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)
        if self.add_time:
            out = out + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlin(out)
        return [out,temb]
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False,add_time=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.temb_proj1 = torch.nn.Linear(512, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.temb_proj2 = torch.nn.Linear(512, planes)
        self.conv3 = conv3x3x3(planes, planes * 4, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm3 = Norm_layer(norm_cfg, planes * 4)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.add_time = add_time
    def forward(self, inp):
        x, temb = inp[0],inp[1]
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)
        if self.add_time:
            out = out + self.temb_proj1(nonlinearity(temb))[:, :, None, None, None]
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.nonlin(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.nonlin(out)
        return [out,temb]
class ResNet(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }
    def __init__(self,
                 depth,
                 in_channels=2,
                 shortcut_type='B',
                 norm_cfg='IN',
                 activation_cfg='relu',
                 weight_std=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 32
        self.conv1 = conv3x3x3(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 32)
        self.nonlin1 = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3x3(32, 32, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, 32)
        self.nonlin2 = Activation_layer(activation_cfg, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 320, layers[3], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
        self.temb_proj = torch.nn.Linear(512,32)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='relu', weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std),
                    Norm_layer(norm_cfg, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std,add_time = True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))
        return nn.Sequential(*layers)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x,t):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        self.layers = []
        x = self.nonlin1(self.norm1(self.conv1(x)))
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.nonlin2(self.norm2(self.conv2(x)))
        self.layers.append(x)
        x,_ = self.layer1([x,temb])
        self.layers.append(x)
        x,_ = self.layer2([x,temb])
        self.layers.append(x)
        x,_ = self.layer3([x,temb])
        self.layers.append(x)
        x,_ = self.layer4([x,temb])
        self.layers.append(x)
        return temb
    def get_layers(self):
        return self.layers
class ResNet_plain(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }
    def __init__(self,
                 depth,
                 in_channels=2,
                 shortcut_type='B',
                 norm_cfg='IN',
                 activation_cfg='relu',
                 weight_std=False):
        super(ResNet_plain, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 32
        self.conv1 = conv3x3x3(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 32)
        self.nonlin1 = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3x3(32, 32, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, 32)
        self.nonlin2 = Activation_layer(activation_cfg, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 320, layers[3], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
        self.temb_proj = torch.nn.Linear(512, 32)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='relu',
                    weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std),
                    Norm_layer(norm_cfg, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample,
                            weight_std=weight_std, add_time=False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))
        return nn.Sequential(*layers)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x, t=None):
        temb = t
        self.layers = []
        x = self.nonlin1(self.norm1(self.conv1(x)))
        if temb is not None:
            x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.nonlin2(self.norm2(self.conv2(x)))
        self.layers.append(x)
        x, _ = self.layer1([x, temb])
        self.layers.append(x)
        x, _ = self.layer2([x, temb])
        self.layers.append(x)
        x, _ = self.layer3([x, temb])
        self.layers.append(x)
        x, _ = self.layer4([x, temb])
        self.layers.append(x)
        return temb
    def get_layers(self):
        return self.layers
class Conditional_ResNet(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }
    def __init__(self,
                 depth,
                 in_channels=2,
                 shortcut_type='B',
                 norm_cfg='IN',
                 activation_cfg='relu',
                 weight_std=False):
        super(Conditional_ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        clip_path = os.environ.get("DIFFRESMEM_CLIP_EMBEDDING", "")
        if clip_path and os.path.isfile(clip_path):
            self.organ_embedding = torch.load(clip_path, map_location="cpu").float()
        else:
            self.organ_embedding = nn.Parameter(torch.zeros(4, 512), requires_grad=False)
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 32
        self.conv1 = conv3x3x3(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 32)
        self.nonlin1 = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3x3(32, 32, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, 32)
        self.nonlin2 = Activation_layer(activation_cfg, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 320, layers[3], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])
        self.temb_proj = torch.nn.Linear(512, 32)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='relu',
                    weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std),
                    Norm_layer(norm_cfg, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample,
                            weight_std=weight_std, add_time=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))
        return nn.Sequential(*layers)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x, t,c_idx):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        organ_embedding = []
        b,c,d,h,w = x.shape
        for i in range(b):
            organ_embedding.append(self.organ_embedding[c_idx[i]].clone().detach())
        organ_embedding = torch.tensor(torch.stack(organ_embedding))
        temb = temb + organ_embedding.to(temb.device)
        self.layers = []
        x = self.nonlin1(self.norm1(self.conv1(x)))
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.nonlin2(self.norm2(self.conv2(x)))
        self.layers.append(x)
        x, _ = self.layer1([x, temb])
        self.layers.append(x)
        x, _ = self.layer2([x, temb])
        self.layers.append(x)
        x, _ = self.layer3([x, temb])
        self.layers.append(x)
        x, _ = self.layer4([x, temb])
        self.layers.append(x)
        return temb
    def get_layers(self):
        return self.layers
class ResNetLight(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }
    def __init__(self,
                 depth,
                 in_channels=2,
                 shortcut_type='B',
                 norm_cfg='IN',
                 activation_cfg='relu',
                 weight_std=False):
        super(ResNetLight, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 32
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 320, layers[3], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='relu',
                    weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std),
                    Norm_layer(norm_cfg, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample,
                            weight_std=weight_std, add_time=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))
        return nn.Sequential(*layers)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x, temb):
        self.clear_layers()
        self.layers.append(x)
        x, _ = self.layer1([x, temb])
        self.layers.append(x)
        x, _ = self.layer2([x, temb])
        self.layers.append(x)
        x, _ = self.layer3([x, temb])
        self.layers.append(x)
        x, _ = self.layer4([x, temb])
        self.layers.append(x)
        return temb
    def get_layers(self):
        return self.layers
    def clear_layers(self):
        self.layers = []
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
def nonlinearity(x):
    return x*torch.sigmoid(x)
