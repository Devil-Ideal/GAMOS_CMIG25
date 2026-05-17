import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import ResCNN2
class Conv3d_wd(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(
            dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
              groups=1, bias=True, weight_std=False):
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)
def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        return nn.BatchNorm3d(inplanes)
    if norm_cfg == 'SyncBN':
        return nn.SyncBatchNorm(inplanes)
    if norm_cfg == 'GN':
        return nn.GroupNorm(16, inplanes)
    if norm_cfg == 'IN':
        return nn.InstanceNorm3d(inplanes, affine=True)
    raise ValueError(f"Unsupported norm_cfg: {norm_cfg}")
def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'relu':
        return nn.ReLU(inplace=inplace)
    if activation_cfg == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=1e-1, inplace=inplace)
    raise ValueError(f"Unsupported activation_cfg: {activation_cfg}")
class ResCNN_group_mem_sparse(nn.Module):
    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', in_channels=2, num_classes=None,
                 weight_std=False, res_depth=None, mem_size=2048, sparse=0.75):
        super().__init__()
        self.num_classes = num_classes
        expansion = 4 if res_depth >= 50 else 1
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.cnn_bottle = nn.Sequential(
            conv3x3x3(320 * expansion, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.shortcut_conv3 = nn.Sequential(
            conv3x3x3(256 * expansion, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.shortcut_conv2 = nn.Sequential(
            conv3x3x3(128 * expansion, 128, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 128),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.shortcut_conv1 = nn.Sequential(
            conv3x3x3(64 * expansion, 64, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 64),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.shortcut_conv0 = nn.Sequential(
            conv3x3x3(32, 32, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 32),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.transposeconv_stage3 = nn.ConvTranspose3d(256, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.stage3_de = ResCNN2.BasicBlock(256, 256, norm_cfg, activation_cfg, weight_std=weight_std, add_time=True)
        self.stage2_de = ResCNN2.BasicBlock(128, 128, norm_cfg, activation_cfg, weight_std=weight_std, add_time=True)
        self.stage1_de = ResCNN2.BasicBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std, add_time=True)
        self.stage0_de = ResCNN2.BasicBlock(32, 32, norm_cfg, activation_cfg, weight_std=weight_std, add_time=True)
        self.backbone = ResCNN2.ResNet(
            depth=res_depth, shortcut_type='B', norm_cfg=norm_cfg,
            activation_cfg=activation_cfg, weight_std=weight_std, in_channels=in_channels,
        )
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(32, 8, kernel_size=1),
        )
        self.group_head = nn.Sequential(
            nn.Conv3d(8 * num_classes, 8 * num_classes, kernel_size=3, padding=1, groups=num_classes),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(8 * num_classes, 8 * num_classes, kernel_size=1, padding=0, groups=num_classes),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(8 * num_classes, num_classes, kernel_size=1, padding=0, groups=num_classes),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.mem = nn.Parameter(torch.FloatTensor(1, 256, mem_size).normal_(0.0, 1.0))
        self.sparse = sparse
        self.att = None
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def memory_retrival2(self, x):
        y = F.instance_norm(x, eps=1e-5)
        b, c, d, h, w = y.shape
        m = self.mem.repeat(b, 1, 1)
        m_key = m.clone().transpose(1, 2)
        y_ = y.view(b, c, -1)
        logits = torch.bmm(m_key, y_) / math.sqrt(c)
        logits2 = F.softmax(logits, dim=1)
        quantiles = torch.quantile(logits2, self.sparse, dim=1, keepdim=True)
        logits3 = logits2 - quantiles
        new_logits = torch.relu(logits3)
        logits4 = F.normalize(new_logits, p=1, dim=1)
        y_new = torch.bmm(m_key.transpose(1, 2), logits4)
        y_new_ = y_new.view(b, c, d, h, w)
        return y_new_, logits4
    def forward(self, inputs_x, time_step, aug=False):
        bs, _, _, _, _ = inputs_x.shape
        temb = self.backbone(inputs_x, time_step)
        layers = self.backbone.get_layers()
        x = self.cnn_bottle(layers[-1])
        del self.att
        f, self.att = self.memory_retrival2(x)
        x = x + f
        if aug:
            new_layers = []
            for layer in layers:
                if not isinstance(layer, torch.Tensor):
                    layer = layer.as_tensor()
                new_layers.append(layer[:bs // 2])
            if not isinstance(x, torch.Tensor):
                x = x.as_tensor()
            x = x[:bs // 2]
            temb = temb[:bs // 2]
        else:
            new_layers = layers
        x = self.transposeconv_stage3(x)
        x = x + self.shortcut_conv3(new_layers[-2])
        x, _ = self.stage3_de([x, temb])
        x = self.transposeconv_stage2(x)
        x = x + self.shortcut_conv2(new_layers[-3])
        x, _ = self.stage2_de([x, temb])
        x = self.transposeconv_stage1(x)
        x = x + self.shortcut_conv1(new_layers[-4])
        x, _ = self.stage1_de([x, temb])
        x = self.transposeconv_stage0(x)
        x = x + self.shortcut_conv0(new_layers[-5])
        x, _ = self.stage0_de([x, temb])
        x = self.precls_conv(x)
        x = self.upsample(x)
        logits_array = []
        for i in range(x.shape[0]):
            head_inputs = x[i].unsqueeze(0).repeat(self.num_classes, 1, 1, 1, 1)
            _, _, d, h, w = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, d, h, w)
            logits = self.group_head(head_inputs)
            logits_array.append(logits.reshape(1, -1, d, h, w))
        return torch.cat(logits_array, dim=0)
class ResUnet_mem_sparse(nn.Module):
    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', in_channels=2, num_classes=None,
                 weight_std=False, deep_supervision=False, res_depth=None, mem_size=2048, sparse=0.75):
        super().__init__()
        self.do_ds = deep_supervision
        self.ResCNN_DeformTR = ResCNN_group_mem_sparse(
            norm_cfg, activation_cfg, in_channels, num_classes, weight_std, res_depth, mem_size, sparse,
        )
        self.conv_op = Conv3d_wd if weight_std else nn.Conv3d
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        elif norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        elif norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
    def forward(self, x, t, image=None, c_idx=None, aug=False):
        if image is not None:
            x = torch.cat([image, x], dim=1)
        return self.ResCNN_DeformTR(x, time_step=t, aug=aug)
