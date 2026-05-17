import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb
__all__ = [
    'Mlp',
    'Attention',
    'TransformerBlock',
    'LayerNorm',
]
class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def b_l_hd__b_h_l_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)
        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    def b_h_l_d__b_l_hd(self, x):
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = self.b_h_l_d__b_l_hd(attned)
        attned = self.to_out(attned)
        return attned
class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def b_l_hd__b_h_l_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)
        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    def b_h_l_d__b_l_hd(self, x):
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x
    def forward(self, x1, x2):
        q = self.to_q(x1)
        k, v = self.to_kv(x2).chunk(2, dim=-1)
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = self.b_h_l_d__b_l_hd(attned)
        attned = self.to_out(attned)
        return attned
class FocusedCrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.,focusing_factor=3,kernel_size=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.focusing_factor = focusing_factor
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dwc = nn.Conv3d(in_channels=dim_head, out_channels=dim_head, kernel_size=kernel_size,
                             groups=dim_head, padding=kernel_size // 2)
        self.proj_drop = nn.Dropout(proj_drop)
    def b_l_hd__b_h_l_d(self, x, heads):
        b, l, n = x.shape
        h = heads
        d = int(n / h)
        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    def b_h_l_d__b_l_hd(self, x):
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x
    def forward(self, x1, x2):
        B, N, C = x1.shape
        q = self.to_q(x1)
        k, v = self.to_kv(x2).chunk(2, dim=-1)
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) ) @ (v) /N
        x = q @ kv * z
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, D,-1).permute(0, 4, 1, 2, 3)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = self.to_out(x)
        return x
        attned = self.b_h_l_d__b_l_hd(attned)
class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop)),
                PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
                ]))
    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
            return x
