from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

for candidate in (CURRENT_DIR, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


import torch
import torch.nn as nn
import torch.nn.functional as F

from Main_Model.CSUA_LDM_Config import (
    DEVICE,
    CSUA_LDM_VAE_CONTENT_CHANNELS,
    CSUA_LDM_DIFFUSION_STEPS,
    CSUA_LDM_DIFFUSION_COND_SCALE,
    CSUA_LDM_DIFFUSION_UNC_SCALE,
)
from CSUA_LDM_Diffusion_block import (
    CondBlock,
    Downsample,
    Normalize,
    ResnetBlock,
    UncBlock,
    SelfAttention,
    ShiftedWindowSelfAttention2D,
    Upsample,
    get_timestep_embedding,
    nonlinearity,
)



class CSUA_LDM_Diffusion_UNet(nn.Module):
    """DDPM UNet with v-prediction and reverse uncertainty heads."""
    def __init__(
        self, 
        ch=96, 
        out_ch=CSUA_LDM_VAE_CONTENT_CHANNELS,
        ch_mult=(1, 2, 3, 4), 
        attn_resolutions=[15, 30, 60], 
        dropout=0.0, 
        resamp_with_conv=True, 
        in_channels=CSUA_LDM_VAE_CONTENT_CHANNELS,
        resolution=60,
        cond_channels=CSUA_LDM_VAE_CONTENT_CHANNELS,
        cond_scale=CSUA_LDM_DIFFUSION_COND_SCALE,
        unc_scale=CSUA_LDM_DIFFUSION_UNC_SCALE,
        window_attn_min_resolution=64,
        attn_base_head_dim=32,
        attn_heads_cap=12,
    ):
        super().__init__()
        self.window_attn_min_resolution = int(window_attn_min_resolution)
        if self.window_attn_min_resolution <= 0:
            raise ValueError("window_attn_min_resolution must be positive.")
        self.attn_base_head_dim = int(attn_base_head_dim)
        self.attn_heads_cap = int(attn_heads_cap)
        if self.attn_base_head_dim <= 0:
            raise ValueError("attn_base_head_dim must be positive.")
        if self.attn_heads_cap <= 0:
            raise ValueError("attn_heads_cap must be positive.")
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = int(out_ch)

        self.cond_channels = cond_channels
        self.cond_scale = float(cond_scale)
        self.unc_scale = float(unc_scale)
        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)
        print(
            "[CondEncoder][Main] "
            "time-aware condition encoder enabled, "
            f"cond_scale={self.cond_scale}, "
            f"unc_scale={self.unc_scale}"
        )

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        # ---- multi-scale condition encoder (y0) ----
        self.cond_in = nn.Conv2d(cond_channels, ch * ch_mult[0], 3, 1, 1)
        self.cond_down = nn.ModuleList()
        for i in range(self.num_resolutions - 1):
            in_c = ch * ch_mult[i]
            out_c = ch * ch_mult[i + 1]
            self.cond_down.append(nn.Sequential(
                CondBlock(in_c, out_c),
                Downsample(out_c, with_conv=True),
            ))

        self.cond_time_down_scale = nn.ModuleList()
        self.cond_time_down_shift = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            out_c = ch * ch_mult[i_level]
            scale_proj = nn.Linear(self.temb_ch, out_c)
            shift_proj = nn.Linear(self.temb_ch, out_c)
            nn.init.zeros_(scale_proj.weight)
            nn.init.zeros_(scale_proj.bias)
            nn.init.zeros_(shift_proj.weight)
            nn.init.zeros_(shift_proj.bias)
            self.cond_time_down_scale.append(scale_proj)
            self.cond_time_down_shift.append(shift_proj)
        self.cond_time_mid_scale = nn.Linear(self.temb_ch, ch * ch_mult[-1])
        self.cond_time_mid_shift = nn.Linear(self.temb_ch, ch * ch_mult[-1])
        nn.init.zeros_(self.cond_time_mid_scale.weight)
        nn.init.zeros_(self.cond_time_mid_scale.bias)
        nn.init.zeros_(self.cond_time_mid_shift.weight)
        nn.init.zeros_(self.cond_time_mid_shift.bias)

        # project cond to match down-path channels
        # block_in at first block: ch * in_ch_mult[i_level]
        # block_in after first block: ch * ch_mult[i_level]
        self.cond_proj_down_in = nn.ModuleList()
        self.cond_proj_down_out = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            in_c = ch * ch_mult[i_level]
            out_c_in = ch * in_ch_mult[i_level]
            out_c_out = ch * ch_mult[i_level]
            self.cond_proj_down_in.append(nn.Conv2d(in_c, out_c_in, kernel_size=1, stride=1, padding=0))
            self.cond_proj_down_out.append(nn.Conv2d(in_c, out_c_out, kernel_size=1, stride=1, padding=0))

        # project cond to match up-path main branch (h) channels at each level start
        self.cond_proj_up = nn.ModuleList()
        up_h_channels = []

        # ---- precision gating for condition modulation ----
        self.prec_in = nn.Conv2d(cond_channels, ch * ch_mult[0], 3, 1, 1)
        self.prec_down = nn.ModuleList()
        for i in range(self.num_resolutions - 1):
            in_c = ch * ch_mult[i]
            out_c = ch * ch_mult[i + 1]
            self.prec_down.append(nn.Sequential(
                UncBlock(in_c, out_c),
                Downsample(out_c, with_conv=True),
            ))

        self.gate_proj = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            in_c = ch * ch_mult[i_level]
            out_c = ch * ch_mult[i_level]
            proj = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.gate_proj.append(proj)


        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_a = nn.ModuleList()
            block_b = nn.ModuleList()
            self_attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            downblocks_num = 2 if curr_res >= self.window_attn_min_resolution else 3
            for i_block in range(downblocks_num):

                block_a.append(ResnetBlock(in_channels=block_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:
                    if i_block == downblocks_num - 1:
                        downblock_base_head_dim = self.attn_base_head_dim
                        downblock_heads = min(max(block_in // downblock_base_head_dim, 1), self.attn_heads_cap)
                        if curr_res >= self.window_attn_min_resolution:
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim, window_size=8, shift_size=0))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim, window_size=8, shift_size=4))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                        if curr_res < self.window_attn_min_resolution:
                            self_attn.append(SelfAttention(in_channels=block_in, heads=downblock_heads, head_dim=downblock_base_head_dim))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                        
            down = nn.Module()
            down.block_a = block_a
            down.block_b = block_b
            down.self_attn = self_attn

            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)


        # middle
        midblock_base_head_dim = self.attn_base_head_dim
        midblock_heads = min(max(block_in // midblock_base_head_dim, 1), self.attn_heads_cap)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.mid.self_attn_1 = SelfAttention(in_channels=block_in, heads=midblock_heads, head_dim=midblock_base_head_dim)
        
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        self.mid.self_attn_2 = SelfAttention(in_channels=block_in, heads=midblock_heads, head_dim=midblock_base_head_dim)
        
        self.mid.block_3 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        up_h_channels = []
        for i_level in reversed(range(self.num_resolutions)):
            up_h_channels.append(block_in)
            block_a = nn.ModuleList()
            block_b = nn.ModuleList()
            self_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            upblocks_num = 3 if curr_res >= self.window_attn_min_resolution else 4
            for i_block in range(upblocks_num):
                
                if i_block == upblocks_num - 1:
                    skip_in = ch * in_ch_mult[i_level]

                block_a.append(ResnetBlock(in_channels=block_in+skip_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout))
                
                block_in = block_out

                if curr_res in attn_resolutions:
                    if i_block == upblocks_num - 1:
                        upblock_base_head_dim = self.attn_base_head_dim
                        upblock_heads = min(max(block_in // upblock_base_head_dim, 1), self.attn_heads_cap)
                        if curr_res >= self.window_attn_min_resolution:
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=upblock_heads, head_dim=upblock_base_head_dim, window_size=8, shift_size=0))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                            self_attn.append(ShiftedWindowSelfAttention2D(in_channels=block_in, heads=upblock_heads, head_dim=upblock_base_head_dim, window_size=8, shift_size=4))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))
                        if curr_res < self.window_attn_min_resolution:
                            self_attn.append(SelfAttention(in_channels=block_in, heads=upblock_heads, head_dim=upblock_base_head_dim))
                            block_b.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_in,
                                                       temb_channels=self.temb_ch,
                                                       dropout=dropout))

            up = nn.Module()
            up.block_a = block_a
            up.block_b = block_b
            up.self_attn = self_attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        up_h_channels = up_h_channels[::-1]
        for i_level in range(self.num_resolutions):
            self.cond_proj_up.append(nn.Conv2d(ch * ch_mult[i_level], up_h_channels[i_level], kernel_size=1, stride=1, padding=0))

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out_v_mean = torch.nn.Conv2d(
            block_in,
            self.out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.var_norm = Normalize(block_in)
        self.conv_out_v_var = torch.nn.Conv2d(
            block_in,
            self.out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.zeros_(self.conv_out_v_mean.weight)
        nn.init.zeros_(self.conv_out_v_mean.bias)
        nn.init.zeros_(self.conv_out_v_var.weight)
        nn.init.constant_(self.conv_out_v_var.bias, -4.0)

    def _variance_from_raw(self, raw_logvar: torch.Tensor):
        # pred_v_var is the variance in v-prediction space, not the direct x0_hat variance.
        # The sampler maps it to latent-space reverse-step uncertainty via DDPM coefficients.
        pred_var = F.softplus(raw_logvar)
        pred_logvar = torch.log(pred_var)
        return pred_var, pred_logvar

    def _apply_time_aware_condition(
        self,
        cond_feat: torch.Tensor,
        temb: torch.Tensor,
        scale_proj: nn.Linear,
        shift_proj: nn.Linear,
    ) -> torch.Tensor:
        gamma = torch.tanh(scale_proj(temb)).to(device=cond_feat.device, dtype=cond_feat.dtype)
        beta = shift_proj(temb).to(device=cond_feat.device, dtype=cond_feat.dtype)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return cond_feat * (1.0 + gamma) + beta

    def _apply_time_aware_condition_stack(
        self,
        cond_feats: list[torch.Tensor],
        temb: torch.Tensor,
    ) -> list[torch.Tensor]:
        time_aware_feats = []
        for i_level, cond_feat in enumerate(cond_feats):
            time_aware_feats.append(
                self._apply_time_aware_condition(
                    cond_feat,
                    temb,
                    self.cond_time_down_scale[i_level],
                    self.cond_time_down_shift[i_level],
                )
            )
        return time_aware_feats

    def forward(self, xt, y0, y0_sigma, timesteps, return_dict: bool = False):
        if y0_sigma is None:
            raise ValueError("y0_sigma must not be None.")

        # timestep embedding
        temb = get_timestep_embedding(timesteps, self.ch)  # [B, ch]
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # multi-scale condition features (y0)
        cond = self.cond_in(y0)
        cond_feats = [cond]
        for down in self.cond_down:
            cond = down(cond)
            cond_feats.append(cond)

        # precision gating: sigma -> log precision
        eps = 1e-8
        log_precision = -2.0 * torch.log(y0_sigma.clamp_min(eps))
        log_precision = log_precision.clamp(-12.0, 12.0)

        prec = self.prec_in(log_precision)
        prec_feats = [prec]
        for down in self.prec_down:
            prec = down(prec)
            prec_feats.append(prec)

        # modulate cond_feats with precision gates
        modulated_cond_feats = []
        for i_level in range(self.num_resolutions):
            cond_level = cond_feats[i_level]
            prec_level = prec_feats[i_level]
            gate = torch.tanh(self.gate_proj[i_level](prec_level))
            confidence_mod = 1.0 + self.unc_scale * gate
            modulated_cond = cond_level * confidence_mod
            modulated_cond_feats.append(modulated_cond)
        cond_mid_source = self.cond_proj_down_out[-1](modulated_cond_feats[-1])
        modulated_cond_feats = self._apply_time_aware_condition_stack(modulated_cond_feats, temb)

        h0 = self.conv_in(xt)
        hs = [h0]
        cond_inject_scale = self.cond_scale

        # downsampling
        for i_level in range(self.num_resolutions):

            downblocks_num = len(self.down[i_level].block_a)
            cond_level = self.cond_proj_down_in[i_level](modulated_cond_feats[i_level])
            for i_block in range(downblocks_num):

                h = hs[-1]
                if i_block == 0:
                    h = h + cond_inject_scale * cond_level
                h = self.down[i_level].block_a[i_block](h, temb)
                
                if i_block == downblocks_num - 1:
                    for idx in range(len(self.down[i_level].self_attn)):
                        if len(self.down[i_level].self_attn) > 0:
                            h = self.down[i_level].self_attn[idx](h)
                        if len(self.down[i_level].block_b) > 0:
                            h = self.down[i_level].block_b[idx](h, temb)

                hs.append(h)
                
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(hs[-1])
                hs.append(h) 

        # middle
        h = hs[-1]
        cond_mid = self._apply_time_aware_condition(
            cond_mid_source,
            temb,
            self.cond_time_mid_scale,
            self.cond_time_mid_shift,
        )
        h = h + cond_inject_scale * cond_mid
        h = self.mid.block_1(h, temb)
        h = self.mid.self_attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.mid.self_attn_2(h)
        h = self.mid.block_3(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            upblocks_num = len(self.up[i_level].block_a)
            cond_level = modulated_cond_feats[i_level]
            for i_block in range(upblocks_num):

                skip = hs.pop()
                if i_block == 0:
                    if h.shape[1] != cond_level.shape[1]:
                        cond_h = self.cond_proj_up[i_level](cond_level)
                    else:
                        cond_h = cond_level
                    h = h + cond_inject_scale * cond_h
                h = self.up[i_level].block_a[i_block](torch.cat([h, skip], dim=1), temb)

                if i_block == upblocks_num - 1:
                    for idx in range(len(self.up[i_level].self_attn)):
                        if len(self.up[i_level].self_attn) > 0:
                            h = self.up[i_level].self_attn[idx](h)
                        if len(self.up[i_level].block_b) > 0:
                            h = self.up[i_level].block_b[idx](h, temb)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)

        pred_v_mean = self.conv_out_v_mean(h)

        var_h = self.var_norm(h)
        var_h = nonlinearity(var_h)

        raw_v_var = self.conv_out_v_var(var_h)
        pred_v_var, pred_v_logvar = self._variance_from_raw(raw_v_var)

        if return_dict:
            return {
                "pred_v_mean": pred_v_mean,
                "pred_v_var": pred_v_var,
                "pred_v_logvar": pred_v_logvar,
                "pred_v_var_raw": raw_v_var,
            }

        return pred_v_mean



if __name__ == "__main__":

    imgs_e = torch.rand((3, CSUA_LDM_VAE_CONTENT_CHANNELS, 64, 64)).to(DEVICE)
    imgs_c = torch.rand((3, CSUA_LDM_VAE_CONTENT_CHANNELS, 64, 64)).to(DEVICE)
    imgs_unc = torch.rand((3, CSUA_LDM_VAE_CONTENT_CHANNELS, 64, 64)).to(DEVICE) * 0.5

    print('imgs_e.shape: ', imgs_e.shape)

    diffusion_model = CSUA_LDM_Diffusion_UNet().to(DEVICE)

    t = torch.randint(
        max(int(CSUA_LDM_DIFFUSION_STEPS), 2), size=(imgs_e.shape[0],), device=DEVICE
    )
    out = diffusion_model(imgs_e, imgs_c, imgs_unc, t)
    print("pred_v_mean shape:", out.shape)

    out_dict = diffusion_model(
        imgs_e, imgs_c, imgs_unc, t, return_dict=True
    )
    print("return_dict keys:", list(out_dict.keys()))
