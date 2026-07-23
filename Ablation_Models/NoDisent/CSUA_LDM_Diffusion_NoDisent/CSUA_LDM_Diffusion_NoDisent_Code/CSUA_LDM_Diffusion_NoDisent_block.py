from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VARIANT_ROOT.parents[1]

for candidate in (CURRENT_DIR, PROJECT_ROOT, VARIANT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4, head_dim=16, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim

        self.norm = Normalize(in_channels)

        # Note: qkv outputs use inner_dim, not in_channels
        self.qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=False)

        # Project outputs back to in_channels for residual addition
        self.proj_out = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1, bias=False)

        self.dropout = dropout

    def forward(self, x):
        b, c, H, W = x.shape
        h_ = self.norm(x)

        qkv = self.qkv(h_)                      # (b, 3*inner_dim, H, W)
        q, k, v = qkv.chunk(3, dim=1)           # each: (b, inner_dim, H, W)

        n = H * W
        # (b, heads, n, head_dim)
        q = q.reshape(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = v.reshape(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )  # (b, heads, n, head_dim)

        out = out.permute(0, 1, 3, 2).reshape(b, self.inner_dim, H, W)  # (b, inner_dim, H, W)
        out = self.proj_out(out)                                        # (b, in_channels, H, W)
        return x + out


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels, heads=4, head_dim=16):
#         super().__init__()
#         assert in_channels % heads == 0
#         self.in_channels = in_channels
#         self.heads = heads
#         self.head_dim = head_dim if head_dim is not None else in_channels // heads

#         self.norm = Normalize(in_channels)
#         self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
#         self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         h_ = self.norm(x)

#         qkv = self.qkv(h_)  # (b, 3c, h, w)
#         q, k, v = qkv.chunk(3, dim=1)

#         # (b, heads, hw, head_dim)
#         q = q.reshape(b, self.heads, self.head_dim, h*w).permute(0,1,3,2)
#         k = k.reshape(b, self.heads, self.head_dim, h*w).permute(0,1,3,2)
#         v = v.reshape(b, self.heads, self.head_dim, h*w).permute(0,1,3,2)

#         # SDPA: (b, heads, hw, head_dim)
#         out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

#         # back to (b, c, h, w)
#         out = out.permute(0,1,3,2).reshape(b, c, h, w)
#         out = self.proj_out(out)
#         return x + out



class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x



class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def get_timestep_embedding(timesteps, embedding_dim):

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


class CondBlock(nn.Module):
    """
    Norm -> SiLU -> Conv(stride=1, padding=1)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = Normalize(in_ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.act(self.norm(x))
        return self.conv(x)


class UncBlock(nn.Module):
    """
    Same structure as CondBlock, dedicated to the uncertainty branch.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = Normalize(in_ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.act(self.norm(x))
        return self.conv(x)


def zero_linear(in_dim, out_dim):
    lin = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class TimestepScaleShift(nn.Module):
    """
    Generate per-channel (scale, shift) from temb
    Output shape: [B, C, 1, 1]
    """
    def __init__(self, temb_dim, channels):
        super().__init__()
        self.proj = zero_linear(temb_dim, channels * 2)

    def forward(self, temb):
        # temb: [B, temb_dim]
        ss = self.proj(temb)                 # [B, 2C]
        scale, shift = ss.chunk(2, dim=1)    # each: [B, C]
        scale = scale[:, :, None, None]      # [B, C, 1, 1]
        shift = shift[:, :, None, None]
        return scale, shift


def window_partition(x, window_size: int):
    """
    x: (B, C, H, W)
    return: windows (B * num_windows, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, C,
               H // window_size, window_size,
               W // window_size, window_size)
    # (B, num_h, num_w, ws, ws, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(B * (H // window_size) * (W // window_size),
                     window_size * window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int, B: int):
    """
    windows: (B * num_windows, window_size*window_size, C)
    return: (B, C, H, W)
    """
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x




def build_shift_attn_mask(Hp, Wp, ws, shift, device):
    # (1, 1, Hp, Wp) used as a region-index map
    img_mask = torch.zeros((1, 1, Hp, Wp), device=device)
    cnt = 0
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1

    # After window partition: (num_win, T, 1) -> (num_win, T)
    mask_windows = window_partition(img_mask, ws).squeeze(-1)

    # Difference mask: 0 for same region, non-zero for different regions
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_win, T, T)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, 0.0)
    # For SDPA: shape is broadcastable to (B*nwin, heads, T, T)
    return attn_mask.unsqueeze(1)  # (num_win, 1, T, T)


class ShiftedWindowSelfAttention2D(nn.Module):
    def __init__(self, in_channels, window_size=8, shift_size=0, heads=4, head_dim=16, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.shift_size = shift_size
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.dropout = dropout

        self.norm = Normalize(in_channels)
        self.qkv = nn.Conv2d(in_channels, 3 * self.inner_dim, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(self.inner_dim, in_channels, 1, bias=False),
            nn.Dropout(dropout)
        )

        # Simple cache; rebuilt when the spatial resolution changes
        self._mask_cache = {}

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        shift = self.shift_size % ws

        # Pad height/width to multiples of the window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x_in = x
        x = self.norm(x)

        # cyclic shift
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Partition feature maps into local windows
        q_w = window_partition(q, ws)
        k_w = window_partition(k, ws)
        v_w = window_partition(v, ws)

        tokens = ws * ws
        q_w = q_w.view(-1, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k_w = k_w.view(-1, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v_w = v_w.view(-1, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_mask = None
        if shift > 0:
            key = (Hp, Wp, ws, shift, x.device.type)
            if key not in self._mask_cache:
                self._mask_cache[key] = build_shift_attn_mask(Hp, Wp, ws, shift, x.device)
            base_mask = self._mask_cache[key]  # (num_win, 1, T, T)

            num_win = (Hp // ws) * (Wp // ws)
            # Repeat across the batch dimension: (B*num_win, 1, T, T)
            attn_mask = base_mask.repeat(B, 1, 1, 1)

        out_w = F.scaled_dot_product_attention(
            q_w, k_w, v_w,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        out_w = out_w.permute(0, 2, 1, 3).contiguous().view(-1, tokens, self.inner_dim)
        out = window_reverse(out_w, ws, Hp, Wp, B)

        # reverse shift
        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(2, 3))

        out = self.proj(out)
        out = x_in + out

        # unpad
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out
