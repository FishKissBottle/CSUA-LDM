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


from CSUA_LDM_NoDisent_Config import (
    CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_ENABLED,
    CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MAX,
    CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MIN,
    CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_RELIABILITY_ALPHA,
    CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_SAVE_MAPS,
    CSUA_LDM_DIFFUSION_NODISENT_SAMPLING_NOISE_SCALE,
    CSUA_LDM_DIFFUSION_NODISENT_UNC_COND_LOGVAR_SOFTEN_LAMBDA,
)
from CSUA_LDM_Diffusion_NoDisent_helpers import extract
from CSUA_LDM_Uncertainty_Guidance import (
    build_guidance_maps,
)
from CSUA_LDM_Utils import sigma_from_variance


class DDPMSampler(nn.Module):
    """Strict v-pred DDPM sampler with uncertainty tracking."""

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

        self.enable_uncertainty_guidance = bool(CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_ENABLED)
        self.guidance_reliability_alpha = float(CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_RELIABILITY_ALPHA)
        self.guidance_noise_gate_min = float(CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MIN)
        self.guidance_noise_gate_max = float(CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MAX)
        self.guidance_save_maps = bool(CSUA_LDM_DIFFUSION_NODISENT_INFERENCE_UNCERT_GUIDANCE_SAVE_MAPS)

    def _autocast_ctx(self):
        if torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _to_ptr_tensor(self, bsz: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((bsz,), device=device, dtype=torch.long)

    @torch.no_grad()
    def _build_guidance(self, input_uncertainty: torch.Tensor, latent_uncertainty: torch.Tensor):
        if not self.enable_uncertainty_guidance:
            return None
        return build_guidance_maps(
            uncertainty_source_map=latent_uncertainty,
            reliability_alpha=self.guidance_reliability_alpha,
            noise_gate_min=self.guidance_noise_gate_min,
            noise_gate_max=self.guidance_noise_gate_max,
        )

    @torch.no_grad()
    def _export_guidance_maps(self, guidance_maps):
        if guidance_maps is None or not self.guidance_save_maps:
            return {}
        info = {
            "guidance_uncertainty_source": guidance_maps["uncertainty_source"],
            "guidance_uncertainty_norm": guidance_maps["uncertainty_norm"],
            "guidance_reliability": guidance_maps["reliability"],
            "guidance_noise_gate": guidance_maps["noise_gate"],
        }
        return info

    def resolve_start_step(self) -> int:
        return max(int(self.T) - 1, 0)

    @torch.no_grad()
    def _parse_model_pred(self, model_out):
        if isinstance(model_out, dict):
            pred_v_mean = model_out.get("pred_v_mean", None)
            pred_v_var = model_out.get("pred_v_var", None)
            pred_v_logvar = model_out.get("pred_v_logvar", None)
        else:
            pred_v_mean = model_out
            pred_v_var = None
            pred_v_logvar = None

        if pred_v_mean is None:
            raise KeyError("Diffusion model output must include `pred_v_mean` or return a tensor.")

        if pred_v_var is None:
            if pred_v_logvar is None:
                pred_v_var = torch.full_like(pred_v_mean, 1e-8)
                pred_v_logvar = torch.log(pred_v_var)
            else:
                pred_v_logvar = torch.clamp(pred_v_logvar, min=-30.0, max=20.0)
                pred_v_var = torch.exp(pred_v_logvar)
        else:
            pred_v_var = torch.clamp(pred_v_var, min=1e-8, max=1e4)
            if pred_v_logvar is None:
                pred_v_logvar = torch.log(pred_v_var)

        return pred_v_mean, pred_v_var, pred_v_logvar

    @torch.no_grad()
    def _predict_v(
        self,
        x_t,
        t,
        y0,
        y0_sigma_sq,
        guidance_maps=None,
    ):
        y0_sigma = sigma_from_variance(
            y0_sigma_sq,
            soften_lambda=CSUA_LDM_DIFFUSION_NODISENT_UNC_COND_LOGVAR_SOFTEN_LAMBDA,
        )
        with self._autocast_ctx():
            out = self.model(
                x_t,
                y0,
                y0_sigma,
                t,
                return_dict=True,
            )

        pred_v_mean, pred_v_var, pred_v_logvar = self._parse_model_pred(out)
        pred_v_mean = torch.nan_to_num(pred_v_mean, nan=0.0, posinf=1e4, neginf=-1e4)
        pred_v_var = torch.nan_to_num(pred_v_var, nan=1e-8, posinf=1e4, neginf=1e-8)
        pred_v_var = torch.clamp(pred_v_var, min=1e-8, max=1e4)
        pred_v_logvar = torch.log(pred_v_var)
        return pred_v_mean, pred_v_var, pred_v_logvar

    @torch.no_grad()
    def _v_to_x0_eps(self, x_t, t, pred_v):
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * pred_v
        eps = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * pred_v
        return x0, eps

    @torch.no_grad()
    def cal_mean_sigma(
        self,
        x_t,
        t,
        y0,
        y0_sigma_sq,
        guidance_maps=None,
    ):
        pred_v, pred_v_var, pred_v_logvar = self._predict_v(
            x_t,
            t,
            y0,
            y0_sigma_sq,
            guidance_maps=guidance_maps,
        )

        x0_hat, _ = self._v_to_x0_eps(x_t, t, pred_v)

        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0_hat
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        sigma = torch.sqrt(extract(self.posterior_variance, t, x_t.shape))

        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)
        beta = extract(self.betas, t, x_t.shape)
        reverse_var = beta.pow(2) * alpha_bar_prev / torch.clamp(
            1.0 - alpha_bar, min=1e-20
        ) * pred_v_var
        reverse_logvar = torch.log(torch.clamp(reverse_var, min=1e-8))

        return mean, sigma, x0_hat, reverse_var, reverse_logvar


    @torch.no_grad()
    def forward(
        self,
        y0,
        x_init,
        y0_sigma_sq,
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
        final_info = None
        last_guidance_export = {}
        latent_uncert_var = y0_sigma_sq.to(device=y0.device, dtype=y0.dtype)
        sampling_noise_scale = float(max(0.0, min(1.0, CSUA_LDM_DIFFUSION_NODISENT_SAMPLING_NOISE_SCALE)))
        sampling_noise_scale_sq = sampling_noise_scale * sampling_noise_scale

        for step_idx, time_step in enumerate(pbar):
            B = x_t.shape[0]
            t = torch.full((B,), time_step, device=x_t.device, dtype=torch.long)
            guidance_maps = self._build_guidance(
                input_uncertainty=y0_sigma_sq,
                latent_uncertainty=latent_uncert_var,
            )
            if time_step == 0:
                pred_v_mean, pred_v_var, pred_v_logvar = self._predict_v(
                    x_t,
                    t,
                    y0,
                    y0_sigma_sq=y0_sigma_sq,
                    guidance_maps=guidance_maps,
                )
                x0_hat, _ = self._v_to_x0_eps(x_t, t, pred_v_mean)
                alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)

                reverse_var = (1.0 - alpha_bar) * pred_v_var
                latent_uncert_var = alpha_bar * latent_uncert_var + reverse_var
                latent_uncert_var = torch.clamp(latent_uncert_var, min=1e-8, max=1e6)

                final_info = {
                    "x0_mean": x0_hat,
                    "reverse_var": reverse_var,
                    "reverse_logvar": torch.log(torch.clamp(reverse_var, min=1e-8)),
                    "latent_var": latent_uncert_var,
                    "latent_logvar": torch.log(torch.clamp(latent_uncert_var, min=1e-8)),
                }
                last_guidance_export = self._export_guidance_maps(guidance_maps)
                x_t = x0_hat
            else:
                mean, sigma, _, reverse_var, reverse_logvar = self.cal_mean_sigma(
                    x_t,
                    t,
                    y0,
                    y0_sigma_sq=y0_sigma_sq,
                    guidance_maps=guidance_maps,
                )
                alpha_t = extract(self.alphas, t, x_t.shape)
                posterior_variance = extract(self.posterior_variance, t, x_t.shape)
                guided_uncert_var = alpha_t * latent_uncert_var + reverse_var
                noise_gate = None
                if guidance_maps is not None:
                    noise_gate = guidance_maps["noise_gate"]
                noise_gate_sq = 1.0 if noise_gate is None else noise_gate.pow(2)
                latent_uncert_var = guided_uncert_var + posterior_variance * sampling_noise_scale_sq * noise_gate_sq
                z = torch.randn_like(x_t)
                sigma_eff = sigma * sampling_noise_scale
                if noise_gate is not None:
                    sigma_eff = sigma_eff * noise_gate
                x_t = mean + sigma_eff * z
                latent_uncert_var = torch.clamp(latent_uncert_var, min=1e-8, max=1e6)
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e4, neginf=-1e4)
                if torch.isnan(x_t).any():
                    raise ValueError("NaN detected in x_prev")
                last_guidance_export = self._export_guidance_maps(guidance_maps)

            if show_pbar:
                pbar.set_postfix({"t": time_step})

        final_info.update(last_guidance_export)
        final_info["x0_sample"] = x_t
        final_info["start_step"] = torch.tensor(
            int(start_step),
            device=x_t.device,
            dtype=torch.long,
        )
        return x_t, final_info
