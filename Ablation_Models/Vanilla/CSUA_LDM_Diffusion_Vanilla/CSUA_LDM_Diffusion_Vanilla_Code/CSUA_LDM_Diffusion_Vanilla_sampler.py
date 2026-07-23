import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VARIANT_ROOT.parents[1]

for candidate in (CURRENT_DIR, PROJECT_ROOT, VARIANT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


from CSUA_LDM_Vanilla_Config import (
    CSUA_LDM_DIFFUSION_VANILLA_SAMPLING_NOISE_SCALE,
)
from CSUA_LDM_Diffusion_Vanilla_helpers import extract


class DDPMSampler(nn.Module):
    """Strict v-pred DDPM sampler for the Vanilla variant (uncertainty-free)."""

    def __init__(
        self,
        model,
        betas: torch.Tensor,
    ):
        super().__init__()
        self.model = model

        T = int(len(betas))
        self.T = T
        betas = betas.float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]]
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(
            1.0 - alphas_cumprod, min=1e-20
        )
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / torch.clamp(
            1.0 - alphas_cumprod, min=1e-20
        )
        posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / torch.clamp(
            1.0 - alphas_cumprod, min=1e-20
        )
        sqrt_alphas_cumprod_prev = torch.sqrt(alphas_cumprod_prev)
        sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - alphas_cumprod_prev)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("sqrt_alphas_cumprod_prev", sqrt_alphas_cumprod_prev)
        self.register_buffer("sqrt_one_minus_alphas_cumprod_prev", sqrt_one_minus_alphas_cumprod_prev)

    def _autocast_ctx(self):
        if torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def resolve_start_step(self) -> int:
        return max(int(self.T) - 1, 0)

    @torch.no_grad()
    def _predict_v(self, x_t, t, y0):
        with self._autocast_ctx():
            pred_v_mean = self.model(
                x_t,
                t,
                y0,
            )

        if pred_v_mean is None:
            raise KeyError("Diffusion model must return `pred_v_mean` for Vanilla sampling.")
        pred_v_mean = torch.nan_to_num(pred_v_mean, nan=0.0, posinf=1e4, neginf=-1e4)
        return pred_v_mean

    @torch.no_grad()
    def _v_to_x0_eps(self, x_t, t, pred_v):
        """Convert v-prediction to x0 and eps using alpha_bar_t."""
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * pred_v
        eps = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * pred_v
        return x0, eps

    @torch.no_grad()
    def cal_mean_sigma(self, x_t, t, y0):
        """Compute strict DDPM posterior mean and sigma for x_{t-1}."""
        pred_v = self._predict_v(
            x_t,
            t,
            y0,
        )

        x0_hat, _ = self._v_to_x0_eps(x_t, t, pred_v)

        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0_hat
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        sigma = torch.sqrt(extract(self.posterior_variance, t, x_t.shape))
        return mean, sigma, x0_hat


    @torch.no_grad()
    def forward(
        self,
        y0,
        x_init,
        start_step=None,
        show_pbar=True,
    ):
        x_t = x_init.to(device=y0.device, dtype=y0.dtype)

        if bool((x_t.abs().sum() == 0).item()):
            raise ValueError("x_init must not be an all-zero tensor.")

        if start_step is None:
            start_step = self.resolve_start_step()
        start_step = max(0, min(self.T - 1, int(start_step)))
        total_steps = start_step + 1
        steps = reversed(range(total_steps))
        pbar = tqdm(steps, total=total_steps, ncols=90) if show_pbar else steps
        sampling_noise_scale = float(max(0.0, min(1.0, CSUA_LDM_DIFFUSION_VANILLA_SAMPLING_NOISE_SCALE)))

        for time_step in pbar:
            B = x_t.shape[0]
            t = torch.full((B,), time_step, device=x_t.device, dtype=torch.long)
            if time_step == 0:
                pred_v_mean = self._predict_v(
                    x_t,
                    t,
                    y0,
                )
                x0_hat, _ = self._v_to_x0_eps(x_t, t, pred_v_mean)
                x_t = x0_hat
            else:
                mean, sigma, _ = self.cal_mean_sigma(
                    x_t,
                    t,
                    y0,
                )
                z = torch.randn_like(x_t)
                x_t = mean + sigma * sampling_noise_scale * z

                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e4, neginf=-1e4)
                if torch.isnan(x_t).any():
                    raise ValueError("NaN detected in x_prev")

            if show_pbar:
                pbar.set_postfix({"t": time_step})

        return x_t
