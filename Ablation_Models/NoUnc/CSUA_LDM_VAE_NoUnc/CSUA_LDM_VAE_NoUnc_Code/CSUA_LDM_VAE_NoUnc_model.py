import sys
from pathlib import Path


VARIANT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VARIANT_ROOT.parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for _search_path in (REPO_ROOT, VARIANT_ROOT, MODULE_DIR):
    search_path_str = str(_search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import torch
import torch.nn as nn

try:
    from CSUA_LDM_VAE_NoUnc.CSUA_LDM_VAE_NoUnc_Code.CSUA_LDM_VAE_NoUnc_block import DownBlock, MidBlock, UpBlock
except ModuleNotFoundError:
    from CSUA_LDM_VAE_NoUnc_block import DownBlock, MidBlock, UpBlock


class LatentAutoEncoder(nn.Module):
    """VAE with explicit Content/Style latent disentanglement (NoUnc version).

    Encoder outputs z_params (mu + logvar), where mu and logvar are each
    split into Content and Style subspaces.

    Decoder fuses Content and Style via separate projection layers before
    the main decoding pipeline. No uncertainty heads or variance prediction.
    """

    def __init__(
        self,
        img_channels: int = 4,
        down_channels: list[int] = [32, 64, 96, 128],
        mid_channels: list[int] = [128, 128],
        num_down_layers: int = 1,
        num_mid_layers: int = 1,
        num_up_layers: int = 1,
        z_channels: int = 8,
        norm_groups: int = 16,
        content_channels: int = 6,
        style_channels: int = 2,
    ):
        super().__init__()

        if img_channels <= 0:
            raise ValueError("`img_channels` must be > 0.")
        if z_channels <= 0:
            raise ValueError("`z_channels` must be > 0.")
        if norm_groups <= 0:
            raise ValueError("`norm_groups` must be > 0.")

        if not isinstance(down_channels, (list, tuple)) or len(down_channels) < 2:
            raise ValueError("`down_channels` must be a list/tuple with at least 2 entries.")
        if not isinstance(mid_channels, (list, tuple)) or len(mid_channels) < 2:
            raise ValueError("`mid_channels` must be a list/tuple with at least 2 entries.")

        self.img_channels = int(img_channels)
        self.down_channels = [int(ch) for ch in down_channels]
        self.mid_channels = [int(ch) for ch in mid_channels]
        if any(ch <= 0 for ch in self.down_channels + self.mid_channels):
            raise ValueError("All entries in `down_channels` and `mid_channels` must be > 0.")

        self.num_down_layers = int(num_down_layers)
        self.num_mid_layers = int(num_mid_layers)
        self.num_up_layers = int(num_up_layers)

        self.z_channels = int(z_channels)
        self.norm_groups = int(norm_groups)
        self.content_channels = int(content_channels)
        self.style_channels = int(style_channels)

        assert self.content_channels + self.style_channels == self.z_channels, (
            f"content_channels ({self.content_channels}) + style_channels ({self.style_channels}) "
            f"must equal z_channels ({self.z_channels})."
        )

        if self.content_channels <= 0 or self.style_channels <= 0:
            raise ValueError(
                f"`content_channels` ({self.content_channels}) and `style_channels` ({self.style_channels}) "
                "must both be > 0."
            )

        self.act = nn.SiLU()

        all_norm_channels = list(self.down_channels) + list(self.mid_channels)
        if any(ch % self.norm_groups != 0 for ch in all_norm_channels):
            raise ValueError("Each channel count in down_channels/mid_channels must be divisible by norm_groups.")

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]

        # ---------------- Encoder ----------------
        self.enc_conv_in = nn.Conv2d(self.img_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.enc_down_blocks = nn.ModuleList(
            [
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    num_layers=self.num_down_layers,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.down_channels) - 1)
            ]
        )

        self.enc_mid_blocks = nn.ModuleList(
            [
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_groups,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        self.enc_norm_out = nn.GroupNorm(self.norm_groups, self.down_channels[-1])
        self.enc_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels * 2, kernel_size=3, padding=1)

        # ---------------- Decoder (Content/Style fusion) ----------------
        self.z_content_proj = nn.Conv2d(self.content_channels, self.down_channels[-1], kernel_size=3, padding=1)
        self.z_style_proj = nn.Conv2d(self.style_channels, self.down_channels[-1], kernel_size=3, padding=1)

        self.dec_mid_blocks = nn.ModuleList(
            [
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i - 1],
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_groups,
                )
                for i in reversed(range(1, len(self.mid_channels)))
            ]
        )

        self.dec_up_blocks = nn.ModuleList(
            [
                UpBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i - 1],
                    num_layers=self.num_up_layers,
                    norm_channels=self.norm_groups,
                )
                for i in reversed(range(1, len(self.down_channels)))
            ]
        )

        self.dec_norm_out = nn.GroupNorm(self.norm_groups, self.down_channels[0])
        self.dec_conv_mean = nn.Conv2d(self.down_channels[0], self.img_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_conv_in(x)

        for blk in self.enc_down_blocks:
            h = blk(h)
        for blk in self.enc_mid_blocks:
            h = blk(h)

        h = self.enc_norm_out(h)
        h = self.act(h)
        z_params = self.enc_conv_out(h)
        return z_params

    def reparameterize(self, z_params: torch.Tensor):
        mu, logvar = torch.chunk(z_params, 2, dim=1)

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + std * eps
        else:
            z = mu
            
        z_content = z[:, :self.content_channels]
        z_style = z[:, self.content_channels:]
        mu_content = mu[:, :self.content_channels]
        mu_style = mu[:, self.content_channels:]
        logvar_content = logvar[:, :self.content_channels]
        logvar_style = logvar[:, self.content_channels:]

        return z_content, z_style, mu_content, mu_style, logvar_content, logvar_style

    def decode(self, z_content: torch.Tensor, z_style: torch.Tensor):
        h = self.z_content_proj(z_content) + self.z_style_proj(z_style)

        for blk in self.dec_mid_blocks:
            h = blk(h)

        for blk in self.dec_up_blocks:
            h = blk(h)

        h = self.dec_norm_out(h)
        h = self.act(h)

        pred_mean = self.dec_conv_mean(h)
        return pred_mean

    def forward(self, x: torch.Tensor):
        z_params = self.encode(x)
        z_content, z_style, _, _, _, _ = self.reparameterize(z_params)
        pred_mean = self.decode(z_content, z_style)
        return pred_mean

if __name__ == '__main__':
    draw_imgs = torch.randn((3, 4, 512, 512))

    vae_model = LatentAutoEncoder()

    pred_mean = vae_model(draw_imgs)

    print('pred_mean.shape: ', pred_mean.shape)
