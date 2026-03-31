# AceFlow v1.0
# Built on top of Ace-Step v1.5
#
# Copyright (C) 2026 Marco Robustini [Marcopter]
#
# This file is part of AceFlow.
# AceFlow is licensed under the GNU General Public License v3.0 or later.
#
# You may redistribute and/or modify this software under the terms
# of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# AceFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import torch

ModelFn = Callable[[torch.Tensor, float], torch.Tensor]

SOLVER_DESCRIPTIONS = {
    "ode": {
        "en": "Deterministic Euler baseline. Fastest and most predictable. Same seed tends to stay closer to the same trajectory.",
        "it": "Baseline Euler deterministico. Più rapido e prevedibile. A parità di seed tende a restare più vicino alla stessa traiettoria.",
    },
    "sde": {
        "en": "Stochastic diffusion path. Re-injects noise during sampling, often giving more variation but less strict repeatability.",
        "it": "Percorso diffusivo stocastico. Reintroduce rumore durante il sampling, spesso con più variazione ma meno ripetibilità stretta.",
    },
    "midpoint": {
        "en": "Second-order midpoint solver. Uses an intermediate evaluation to smooth the update. Slower than ODE, usually steadier.",
        "it": "Solver midpoint del secondo ordine. Usa una valutazione intermedia per rendere l'aggiornamento più regolare. Più lento di ODE, di solito più stabile.",
    },
    "heun": {
        "en": "Second-order predictor-corrector solver. Tries to correct the first estimate before the final step. Often stable and controlled.",
        "it": "Solver predictor-corrector del secondo ordine. Prova a correggere la prima stima prima del passo finale. Spesso stabile e controllato.",
    },
    "rk4": {
        "en": "Classical fourth-order Runge-Kutta. Most expensive among these options, with multiple evaluations per step for a more accurate trajectory.",
        "it": "Runge-Kutta classico del quarto ordine. È il più costoso tra queste opzioni, con più valutazioni per passo per una traiettoria più accurata.",
    },
    "dpm_pp_2m": {
        "en": "Second-order multistep DPM++ style solver. After the first step it reuses past velocity information, often giving a good quality/speed balance.",
        "it": "Solver multistep del secondo ordine in stile DPM++. Dopo il primo passo riusa l'informazione di velocità precedente, spesso con un buon equilibrio tra qualità e velocità.",
    },
}

VALID_INFER_METHODS = ("ode", "sde", "midpoint", "heun", "rk4", "dpm_pp_2m")
_MULTI_EVAL_METHODS = {"midpoint", "heun", "rk4"}
_PATCH_LOG_PREFIX = "[AceFlow InferPatch]"
_PATCH_TARGETS = (
    ("acestep.models.base.modeling_acestep_v15_base", "base", "quality", False),
    ("acestep.models.sft.modeling_acestep_v15_base", "sft", "quality", True),
    ("acestep.models.turbo.modeling_acestep_v15_turbo", "turbo", "turbo", True),
)


def _dt_tensor(dt: float, bsz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((bsz, 1, 1), dt, device=device, dtype=dtype)


def euler_step(model_fn: ModelFn, xt: torch.Tensor, t_curr: float, t_next: float, **_kw) -> Tuple[torch.Tensor, torch.Tensor]:
    vt = model_fn(xt, t_curr)
    dt = _dt_tensor(t_curr - t_next, xt.shape[0], xt.device, xt.dtype)
    return xt - vt * dt, vt


def midpoint_step(model_fn: ModelFn, xt: torch.Tensor, t_curr: float, t_next: float, **_kw) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = t_curr - t_next
    half_dt = _dt_tensor(dt / 2.0, xt.shape[0], xt.device, xt.dtype)
    full_dt = _dt_tensor(dt, xt.shape[0], xt.device, xt.dtype)
    v1 = model_fn(xt, t_curr)
    x_mid = xt - v1 * half_dt
    v_mid = model_fn(x_mid, t_curr - dt / 2.0)
    return xt - v_mid * full_dt, v_mid


def heun_step(model_fn: ModelFn, xt: torch.Tensor, t_curr: float, t_next: float, **_kw) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = t_curr - t_next
    dt_t = _dt_tensor(dt, xt.shape[0], xt.device, xt.dtype)
    v1 = model_fn(xt, t_curr)
    x_pred = xt - v1 * dt_t
    v2 = model_fn(x_pred, t_next)
    v_avg = 0.5 * (v1 + v2)
    return xt - v_avg * dt_t, v_avg


def rk4_step(model_fn: ModelFn, xt: torch.Tensor, t_curr: float, t_next: float, **_kw) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = t_curr - t_next
    half_dt = _dt_tensor(dt / 2.0, xt.shape[0], xt.device, xt.dtype)
    full_dt = _dt_tensor(dt, xt.shape[0], xt.device, xt.dtype)
    t_mid = t_curr - dt / 2.0
    k1 = model_fn(xt, t_curr)
    k2 = model_fn(xt - k1 * half_dt, t_mid)
    k3 = model_fn(xt - k2 * half_dt, t_mid)
    k4 = model_fn(xt - k3 * full_dt, t_next)
    v_eff = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return xt - v_eff * full_dt, v_eff


def dpm_pp_2m_step(
    model_fn: ModelFn,
    xt: torch.Tensor,
    t_curr: float,
    t_next: float,
    prev_vt: Optional[torch.Tensor] = None,
    **_kw,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vt = model_fn(xt, t_curr)
    dt = t_curr - t_next
    dt_t = _dt_tensor(dt, xt.shape[0], xt.device, xt.dtype)
    if prev_vt is None:
        return xt - vt * dt_t, vt
    v_combined = 1.5 * vt - 0.5 * prev_vt
    return xt - v_combined * dt_t, vt


SOLVER_REGISTRY: Dict[str, Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = {
    "euler": euler_step,
    "ode": euler_step,
    "midpoint": midpoint_step,
    "heun": heun_step,
    "rk4": rk4_step,
    "dpm_pp_2m": dpm_pp_2m_step,
}


def get_infer_method_descriptions() -> Dict[str, Dict[str, str]]:
    return {key: {"en": str(value.get("en") or ""), "it": str(value.get("it") or "")} for key, value in SOLVER_DESCRIPTIONS.items()}


def get_runtime_infer_methods(*, use_mlx_dit: bool, patch_installed: bool) -> list[str]:
    if use_mlx_dit:
        return ["ode", "sde"]
    if patch_installed:
        return list(VALID_INFER_METHODS)
    return ["ode", "sde"]


def normalize_infer_method_request(value: str, *, use_mlx_dit: bool, patch_installed: bool) -> tuple[str, str]:
    requested = str(value or "ode").strip().lower() or "ode"
    if requested not in VALID_INFER_METHODS:
        return "ode", "invalid"
    if use_mlx_dit and requested not in {"ode", "sde"}:
        return "ode", "mlx_fallback"
    if not patch_installed and requested not in {"ode", "sde"}:
        return "ode", "patch_missing"
    return requested, "ok"


def _emit_runtime_line(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    try:
        print(text, flush=True)
    except Exception:
        pass
    try:
        logging.getLogger("acestep.ui.aceflow").info(text)
    except Exception:
        pass


def _emit_generation_log(infer_method: str, solver_active: bool, use_kv_cache: bool, variant: str) -> None:
    solver_name = infer_method if solver_active else ("sde" if infer_method == "sde" else "ode")
    cache_name = "on" if use_kv_cache else "off"
    _emit_runtime_line(f"{_PATCH_LOG_PREFIX} active variant={variant} infer_method={infer_method} solver={solver_name} kv_cache={cache_name}")


def _normalize_method(value: str) -> str:
    infer_method = str(value or "ode").strip().lower() or "ode"
    return infer_method if infer_method in VALID_INFER_METHODS else "ode"


def _import_module(module_name: str):
    return importlib.import_module(module_name)


def _quality_generate_audio_impl(
    *,
    self: Any,
    variant: str,
    supports_timesteps: bool,
    module_globals: Dict[str, Any],
    text_hidden_states: torch.FloatTensor,
    text_attention_mask: torch.FloatTensor,
    lyric_hidden_states: torch.FloatTensor,
    lyric_attention_mask: torch.FloatTensor,
    refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
    refer_audio_order_mask: torch.LongTensor,
    src_latents: torch.FloatTensor,
    chunk_masks: torch.FloatTensor,
    is_covers: torch.Tensor,
    silence_latent: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    infer_method: str = "ode",
    use_cache: bool = True,
    infer_steps: int = 30,
    diffusion_guidance_sale: float = 7.0,
    audio_cover_strength: float = 1.0,
    non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
    non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
    audio_codes: Optional[torch.FloatTensor] = None,
    use_progress_bar: bool = True,
    use_adg: bool = False,
    shift: float = 1.0,
    timesteps: Optional[torch.Tensor] = None,
    cover_noise_strength: float = 0.0,
    repaint_mask: Optional[torch.Tensor] = None,
    clean_src_latents: Optional[torch.FloatTensor] = None,
    repaint_crossfade_frames: int = 10,
    repaint_injection_ratio: float = 0.5,
) -> Dict[str, Any]:
    time_mod = module_globals["time"]
    tqdm_fn = module_globals["tqdm"]
    EncoderDecoderCache = module_globals["EncoderDecoderCache"]
    DynamicCache = module_globals["DynamicCache"]
    MomentumBuffer = module_globals["MomentumBuffer"]
    apg_forward = module_globals["apg_forward"]
    adg_forward = module_globals["adg_forward"]
    logger = module_globals["logger"]
    repaint_step = module_globals["_repaint_step_injection"]
    repaint_blend = module_globals["_repaint_boundary_blend"]

    infer_method = _normalize_method(infer_method)

    if attention_mask is None:
        latent_length = src_latents.shape[1]
        attention_mask = torch.ones(src_latents.shape[0], latent_length, device=src_latents.device, dtype=src_latents.dtype)

    time_costs: Dict[str, Any] = {}
    start_time = time_mod.time()
    total_start_time = start_time
    encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask=refer_audio_order_mask,
        hidden_states=src_latents,
        attention_mask=attention_mask,
        silence_latent=silence_latent,
        src_latents=src_latents,
        chunk_masks=chunk_masks,
        is_covers=is_covers,
        precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
        audio_codes=audio_codes,
    )

    encoder_hidden_states_non_cover = None
    encoder_attention_mask_non_cover = None
    context_latents_non_cover = None
    if audio_cover_strength < 1.0:
        non_is_covers = torch.zeros_like(is_covers, device=is_covers.device, dtype=is_covers.dtype)
        silence_latent_expanded = silence_latent[:, :src_latents.shape[1], :].expand(src_latents.shape[0], -1, -1)
        encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = self.prepare_condition(
            text_hidden_states=non_cover_text_hidden_states,
            text_attention_mask=non_cover_text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_latent_expanded,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=silence_latent_expanded,
            chunk_masks=chunk_masks,
            is_covers=non_is_covers,
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )

    end_time = time_mod.time()
    time_costs["encoder_time_cost"] = end_time - start_time
    start_time = end_time

    device, dtype = context_latents.device, context_latents.dtype
    if supports_timesteps and timesteps is not None:
        t = timesteps.to(device=device, dtype=dtype)
        infer_steps = len(t) - 1
    else:
        t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
        if shift != 1.0:
            t = shift * t / (1 + (shift - 1) * t)

    cover_steps = int(infer_steps * audio_cover_strength)
    iterator = tqdm_fn(zip(t[:-1], t[1:]), total=infer_steps) if use_progress_bar else zip(t[:-1], t[1:])

    noise = self.prepare_noise(context_latents, seed)
    bsz = context_latents.shape[0]
    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    momentum_buffer = MomentumBuffer()

    if cover_noise_strength > 0.0:
        effective_noise_level = 1.0 - cover_noise_strength
        t_values = t[:-1].tolist()
        nearest_t = min(t_values, key=lambda x: abs(x - effective_noise_level))
        start_idx = t_values.index(nearest_t)
        xt = self.renoise(src_latents, nearest_t, noise)
        t = t[start_idx:]
        infer_steps = len(t) - 1
        cover_steps = int(infer_steps * audio_cover_strength)
        iterator = tqdm_fn(zip(t[:-1], t[1:]), total=infer_steps) if use_progress_bar else zip(t[:-1], t[1:])
        logger.info(
            f"[generate_audio] Cover mode: cover_noise_strength={cover_noise_strength}, "
            f"effective_noise_level={effective_noise_level:.4f}, nearest_t={nearest_t:.4f}, "
            f"remaining_steps={infer_steps}"
        )
    else:
        xt = noise

    do_cfg_guidance = diffusion_guidance_sale > 1.0
    if do_cfg_guidance:
        encoder_hidden_states = torch.cat([encoder_hidden_states, self.null_condition_emb.expand_as(encoder_hidden_states)], dim=0)
        encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_attention_mask], dim=0)
        context_latents = torch.cat([context_latents, context_latents], dim=0)
        attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

    solver_fn = SOLVER_REGISTRY.get(infer_method)
    solver_active = infer_method not in {"ode", "sde"} and solver_fn is not None
    use_kv_cache = bool(use_cache) and infer_method not in _MULTI_EVAL_METHODS
    if not use_kv_cache:
        past_key_values = None

    _emit_generation_log(infer_method, solver_active, use_kv_cache, variant)

    def model_fn(x_in: torch.Tensor, t_scalar: float) -> torch.Tensor:
        nonlocal past_key_values
        x = torch.cat([x_in, x_in], dim=0) if do_cfg_guidance else x_in
        t_vec = t_scalar * torch.ones((x.shape[0],), device=device, dtype=dtype)
        out = self.decoder(
            hidden_states=x,
            timestep=t_vec,
            timestep_r=t_vec,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=use_kv_cache,
            past_key_values=past_key_values if use_kv_cache else None,
        )
        vt_raw = out[0]
        if use_kv_cache:
            past_key_values = out[1]
        if do_cfg_guidance:
            pred_cond, pred_null_cond = vt_raw.chunk(2)
            if cfg_interval_start <= t_scalar <= cfg_interval_end:
                if not use_adg:
                    return apg_forward(
                        pred_cond=pred_cond,
                        pred_uncond=pred_null_cond,
                        guidance_scale=diffusion_guidance_sale,
                        momentum_buffer=momentum_buffer,
                        dims=[1],
                    )
                return adg_forward(
                    latents=x_in,
                    noise_pred_cond=pred_cond,
                    noise_pred_uncond=pred_null_cond,
                    sigma=t_scalar,
                    guidance_scale=diffusion_guidance_sale,
                )
            return pred_cond
        return vt_raw

    prev_vt = None
    switched_to_non_cover = False
    with torch.no_grad():
        for step_idx, (t_curr, t_prev) in enumerate(iterator):
            if step_idx >= cover_steps and not switched_to_non_cover:
                switched_to_non_cover = True
                if do_cfg_guidance:
                    encoder_hidden_states_non_cover = torch.cat([
                        encoder_hidden_states_non_cover,
                        self.null_condition_emb.expand_as(encoder_hidden_states_non_cover),
                    ], dim=0)
                    encoder_attention_mask_non_cover = torch.cat([
                        encoder_attention_mask_non_cover,
                        encoder_attention_mask_non_cover,
                    ], dim=0)
                    context_latents_non_cover = torch.cat([
                        context_latents_non_cover,
                        context_latents_non_cover,
                    ], dim=0)
                encoder_hidden_states = encoder_hidden_states_non_cover
                encoder_attention_mask = encoder_attention_mask_non_cover
                context_latents = context_latents_non_cover
                if use_kv_cache:
                    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

            t_curr_val = t_curr.item() if isinstance(t_curr, torch.Tensor) else float(t_curr)
            t_prev_val = t_prev.item() if isinstance(t_prev, torch.Tensor) else float(t_prev)

            if infer_method == "sde":
                vt = model_fn(xt, t_curr_val)
                t_curr_bsz = t_curr * torch.ones((bsz,), device=device, dtype=dtype)
                pred_clean = self.get_x0_from_noise(xt, vt, t_curr_bsz)
                next_timestep = 1.0 - (float(step_idx + 1) / infer_steps)
                xt = self.renoise(pred_clean, next_timestep)
                t_after_step = next_timestep
            elif solver_active:
                xt, step_vt = solver_fn(model_fn, xt, t_curr_val, t_prev_val, prev_vt=prev_vt)
                prev_vt = step_vt
                t_after_step = t_prev_val
            else:
                vt = model_fn(xt, t_curr_val)
                dt = t_curr - t_prev
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * dt_tensor
                t_after_step = t_prev_val

            injection_cutoff = round(repaint_injection_ratio * infer_steps)
            if repaint_mask is not None and clean_src_latents is not None and step_idx < injection_cutoff:
                xt = repaint_step(xt, clean_src_latents, repaint_mask, t_after_step, noise)

    x_gen = xt
    if repaint_mask is not None and clean_src_latents is not None and repaint_crossfade_frames > 0:
        x_gen = repaint_blend(x_gen, clean_src_latents, repaint_mask, repaint_crossfade_frames)

    end_time = time_mod.time()
    time_costs["diffusion_time_cost"] = end_time - start_time
    time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / infer_steps
    time_costs["total_time_cost"] = end_time - total_start_time
    return {"target_latents": x_gen, "time_costs": time_costs}


def _make_quality_patched_fn(original_fn: Callable[..., Any], *, variant: str, module_globals: Dict[str, Any], supports_timesteps: bool) -> Callable[..., Any]:
    if supports_timesteps:
        @wraps(original_fn)
        def patched(
            self,
            text_hidden_states: torch.FloatTensor,
            text_attention_mask: torch.FloatTensor,
            lyric_hidden_states: torch.FloatTensor,
            lyric_attention_mask: torch.FloatTensor,
            refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
            refer_audio_order_mask: torch.LongTensor,
            src_latents: torch.FloatTensor,
            chunk_masks: torch.FloatTensor,
            is_covers: torch.Tensor,
            silence_latent: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            seed: Optional[int] = None,
            infer_method: str = "ode",
            use_cache: bool = True,
            infer_steps: int = 30,
            diffusion_guidance_sale: float = 7.0,
            audio_cover_strength: float = 1.0,
            non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
            non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
            cfg_interval_start: float = 0.0,
            cfg_interval_end: float = 1.0,
            precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
            audio_codes: Optional[torch.FloatTensor] = None,
            use_progress_bar: bool = True,
            use_adg: bool = False,
            shift: float = 1.0,
            timesteps: Optional[torch.Tensor] = None,
            cover_noise_strength: float = 0.0,
            repaint_mask: Optional[torch.Tensor] = None,
            clean_src_latents: Optional[torch.FloatTensor] = None,
            repaint_crossfade_frames: int = 10,
            repaint_injection_ratio: float = 0.5,
            **kwargs,
        ):
            return _quality_generate_audio_impl(
                self=self,
                variant=variant,
                supports_timesteps=True,
                module_globals=module_globals,
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                silence_latent=silence_latent,
                attention_mask=attention_mask,
                seed=seed,
                infer_method=infer_method,
                use_cache=use_cache,
                infer_steps=infer_steps,
                diffusion_guidance_sale=diffusion_guidance_sale,
                audio_cover_strength=audio_cover_strength,
                non_cover_text_hidden_states=non_cover_text_hidden_states,
                non_cover_text_attention_mask=non_cover_text_attention_mask,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
                audio_codes=audio_codes,
                use_progress_bar=use_progress_bar,
                use_adg=use_adg,
                shift=shift,
                timesteps=timesteps,
                cover_noise_strength=cover_noise_strength,
                repaint_mask=repaint_mask,
                clean_src_latents=clean_src_latents,
                repaint_crossfade_frames=repaint_crossfade_frames,
                repaint_injection_ratio=repaint_injection_ratio,
            )
    else:
        @wraps(original_fn)
        def patched(
            self,
            text_hidden_states: torch.FloatTensor,
            text_attention_mask: torch.FloatTensor,
            lyric_hidden_states: torch.FloatTensor,
            lyric_attention_mask: torch.FloatTensor,
            refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
            refer_audio_order_mask: torch.LongTensor,
            src_latents: torch.FloatTensor,
            chunk_masks: torch.FloatTensor,
            is_covers: torch.Tensor,
            silence_latent: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            seed: Optional[int] = None,
            infer_method: str = "ode",
            use_cache: bool = True,
            infer_steps: int = 30,
            diffusion_guidance_sale: float = 7.0,
            audio_cover_strength: float = 1.0,
            non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
            non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
            cfg_interval_start: float = 0.0,
            cfg_interval_end: float = 1.0,
            precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
            audio_codes: Optional[torch.FloatTensor] = None,
            use_progress_bar: bool = True,
            use_adg: bool = False,
            shift: float = 1.0,
            cover_noise_strength: float = 0.0,
            repaint_mask: Optional[torch.Tensor] = None,
            clean_src_latents: Optional[torch.FloatTensor] = None,
            repaint_crossfade_frames: int = 10,
            repaint_injection_ratio: float = 0.5,
            **kwargs,
        ):
            return _quality_generate_audio_impl(
                self=self,
                variant=variant,
                supports_timesteps=False,
                module_globals=module_globals,
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                silence_latent=silence_latent,
                attention_mask=attention_mask,
                seed=seed,
                infer_method=infer_method,
                use_cache=use_cache,
                infer_steps=infer_steps,
                diffusion_guidance_sale=diffusion_guidance_sale,
                audio_cover_strength=audio_cover_strength,
                non_cover_text_hidden_states=non_cover_text_hidden_states,
                non_cover_text_attention_mask=non_cover_text_attention_mask,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
                audio_codes=audio_codes,
                use_progress_bar=use_progress_bar,
                use_adg=use_adg,
                shift=shift,
                timesteps=None,
                cover_noise_strength=cover_noise_strength,
                repaint_mask=repaint_mask,
                clean_src_latents=clean_src_latents,
                repaint_crossfade_frames=repaint_crossfade_frames,
                repaint_injection_ratio=repaint_injection_ratio,
            )
    patched.__aceflow_infer_patch__ = True
    patched.__aceflow_infer_patch_variant__ = variant
    return patched


def _turbo_generate_audio_impl(
    *,
    self: Any,
    module_globals: Dict[str, Any],
    text_hidden_states: torch.FloatTensor,
    text_attention_mask: torch.FloatTensor,
    lyric_hidden_states: torch.FloatTensor,
    lyric_attention_mask: torch.FloatTensor,
    refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
    refer_audio_order_mask: torch.LongTensor,
    src_latents: torch.FloatTensor,
    chunk_masks: torch.FloatTensor,
    is_covers: torch.Tensor,
    silence_latent: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    fix_nfe: int = 8,
    infer_method: str = "ode",
    use_cache: bool = True,
    audio_cover_strength: float = 1.0,
    non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
    non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
    precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
    audio_codes: Optional[torch.FloatTensor] = None,
    shift: float = 3.0,
    timesteps: Optional[torch.Tensor] = None,
    cover_noise_strength: float = 0.0,
    repaint_mask: Optional[torch.Tensor] = None,
    clean_src_latents: Optional[torch.FloatTensor] = None,
    repaint_crossfade_frames: int = 10,
    repaint_injection_ratio: float = 0.5,
) -> Dict[str, Any]:
    time_mod = module_globals["time"]
    EncoderDecoderCache = module_globals["EncoderDecoderCache"]
    DynamicCache = module_globals["DynamicCache"]
    logger = module_globals["logger"]
    repaint_step = module_globals["_repaint_step_injection"]
    repaint_blend = module_globals["_repaint_boundary_blend"]

    infer_method = _normalize_method(infer_method)

    valid_shifts = [1.0, 2.0, 3.0]
    valid_timesteps = [
        1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
        0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
        0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
        0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
    ]
    shift_timesteps = {
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
        3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
    }

    t_schedule_list = None
    if timesteps is not None:
        timesteps_list = timesteps.tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps)
        while timesteps_list and timesteps_list[-1] == 0:
            timesteps_list.pop()
        if len(timesteps_list) < 1:
            logger.warning(f"timesteps length is too short after removing trailing zeros, using default shift={shift}")
        elif len(timesteps_list) > 20:
            logger.warning(f"timesteps length={len(timesteps_list)} exceeds maximum 20, truncating to 20")
            timesteps_list = timesteps_list[:20]
            t_schedule_list = timesteps_list
        else:
            t_schedule_list = timesteps_list
        if t_schedule_list is not None:
            original_timesteps = list(t_schedule_list)
            mapped_timesteps = [min(valid_timesteps, key=lambda x: abs(x - value)) for value in t_schedule_list]
            if original_timesteps != mapped_timesteps:
                logger.warning(f"timesteps mapped to nearest valid values: {original_timesteps} -> {mapped_timesteps}")
            t_schedule_list = mapped_timesteps

    if t_schedule_list is None:
        original_shift = shift
        shift = min(valid_shifts, key=lambda x: abs(x - shift))
        if original_shift != shift:
            logger.warning(f"shift={original_shift} not supported, rounded to nearest valid shift={shift}")
        t_schedule_list = shift_timesteps[shift]

    if attention_mask is None:
        latent_length = src_latents.shape[1]
        attention_mask = torch.ones(src_latents.shape[0], latent_length, device=src_latents.device, dtype=src_latents.dtype)

    time_costs: Dict[str, Any] = {}
    start_time = time_mod.time()
    total_start_time = start_time
    encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask=refer_audio_order_mask,
        hidden_states=src_latents,
        attention_mask=attention_mask,
        silence_latent=silence_latent,
        src_latents=src_latents,
        chunk_masks=chunk_masks,
        is_covers=is_covers,
        precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
        audio_codes=audio_codes,
    )

    encoder_hidden_states_non_cover = None
    encoder_attention_mask_non_cover = None
    context_latents_non_cover = None
    if audio_cover_strength < 1.0:
        non_is_covers = torch.zeros_like(is_covers, device=is_covers.device, dtype=is_covers.dtype)
        silence_latent_expanded = silence_latent[:, :src_latents.shape[1], :].expand(src_latents.shape[0], -1, -1)
        encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = self.prepare_condition(
            text_hidden_states=non_cover_text_hidden_states,
            text_attention_mask=non_cover_text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_latent_expanded,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=silence_latent_expanded,
            chunk_masks=chunk_masks,
            is_covers=non_is_covers,
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )

    end_time = time_mod.time()
    time_costs["encoder_time_cost"] = end_time - start_time
    start_time = end_time

    noise = self.prepare_noise(context_latents, seed)
    bsz, device, dtype = context_latents.shape[0], context_latents.device, context_latents.dtype
    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

    if cover_noise_strength > 0.0:
        effective_noise_level = 1.0 - cover_noise_strength
        nearest_t = min(t_schedule_list, key=lambda x: abs(x - effective_noise_level))
        xt = self.renoise(src_latents, nearest_t, noise)
        start_idx = t_schedule_list.index(nearest_t)
        t_schedule_list = t_schedule_list[start_idx:]
        logger.info(
            f"[generate_audio] Cover mode: cover_noise_strength={cover_noise_strength}, "
            f"effective_noise_level={effective_noise_level:.4f}, nearest_t={nearest_t:.4f}, "
            f"remaining_steps={len(t_schedule_list)}"
        )
    else:
        xt = noise

    t_schedule = torch.tensor(t_schedule_list, device=device, dtype=dtype)
    num_steps = len(t_schedule)
    cover_steps = int(num_steps * audio_cover_strength)
    switched_to_non_cover = False
    solver_fn = SOLVER_REGISTRY.get(infer_method)
    solver_active = infer_method not in {"ode", "sde"} and solver_fn is not None
    use_kv_cache = bool(use_cache) and infer_method not in _MULTI_EVAL_METHODS
    if not use_kv_cache:
        past_key_values = None

    _emit_generation_log(infer_method, solver_active, use_kv_cache, "turbo")

    def model_fn(x_in: torch.Tensor, t_scalar: float) -> torch.Tensor:
        nonlocal past_key_values
        t_vec = t_scalar * torch.ones((bsz,), device=device, dtype=dtype)
        with torch.no_grad():
            out = self.decoder(
                hidden_states=x_in,
                timestep=t_vec,
                timestep_r=t_vec,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=use_kv_cache,
                past_key_values=past_key_values if use_kv_cache else None,
            )
        if use_kv_cache:
            past_key_values = out[1]
        return out[0]

    prev_vt = None
    for step_idx in range(num_steps):
        current_timestep = t_schedule[step_idx].item()
        t_curr_tensor = current_timestep * torch.ones((bsz,), device=device, dtype=dtype)

        if step_idx >= cover_steps and not switched_to_non_cover:
            switched_to_non_cover = True
            encoder_hidden_states = encoder_hidden_states_non_cover
            encoder_attention_mask = encoder_attention_mask_non_cover
            context_latents = context_latents_non_cover
            if use_kv_cache:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        if step_idx == num_steps - 1:
            vt = model_fn(xt, current_timestep)
            xt = self.get_x0_from_noise(xt, vt, t_curr_tensor)
            break

        next_timestep = t_schedule[step_idx + 1].item()
        if infer_method == "sde":
            vt = model_fn(xt, current_timestep)
            pred_clean = self.get_x0_from_noise(xt, vt, t_curr_tensor)
            xt = self.renoise(pred_clean, next_timestep)
            t_after_step = next_timestep
        elif solver_active:
            xt, step_vt = solver_fn(model_fn, xt, current_timestep, next_timestep, prev_vt=prev_vt)
            prev_vt = step_vt
            t_after_step = next_timestep
        else:
            vt = model_fn(xt, current_timestep)
            dt = current_timestep - next_timestep
            dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
            xt = xt - vt * dt_tensor
            t_after_step = next_timestep

        injection_cutoff = round(repaint_injection_ratio * num_steps)
        if repaint_mask is not None and clean_src_latents is not None and step_idx < injection_cutoff:
            xt = repaint_step(xt, clean_src_latents, repaint_mask, t_after_step, noise)

    x_gen = xt
    if repaint_mask is not None and clean_src_latents is not None and repaint_crossfade_frames > 0:
        x_gen = repaint_blend(x_gen, clean_src_latents, repaint_mask, repaint_crossfade_frames)

    end_time = time_mod.time()
    time_costs["diffusion_time_cost"] = end_time - start_time
    time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / num_steps
    time_costs["total_time_cost"] = end_time - total_start_time
    return {"target_latents": x_gen, "time_costs": time_costs}


def _make_turbo_patched_fn(original_fn: Callable[..., Any], *, module_globals: Dict[str, Any]) -> Callable[..., Any]:
    @wraps(original_fn)
    def patched(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.FloatTensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.FloatTensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.LongTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.FloatTensor,
        is_covers: torch.Tensor,
        silence_latent: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        fix_nfe: int = 8,
        infer_method: str = "ode",
        use_cache: bool = True,
        audio_cover_strength: float = 1.0,
        non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
        non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: Optional[torch.FloatTensor] = None,
        shift: float = 3.0,
        timesteps: Optional[torch.Tensor] = None,
        cover_noise_strength: float = 0.0,
        repaint_mask: Optional[torch.Tensor] = None,
        clean_src_latents: Optional[torch.FloatTensor] = None,
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        **kwargs,
    ):
        return _turbo_generate_audio_impl(
            self=self,
            module_globals=module_globals,
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=silence_latent,
            attention_mask=attention_mask,
            seed=seed,
            fix_nfe=fix_nfe,
            infer_method=infer_method,
            use_cache=use_cache,
            audio_cover_strength=audio_cover_strength,
            non_cover_text_hidden_states=non_cover_text_hidden_states,
            non_cover_text_attention_mask=non_cover_text_attention_mask,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes,
            shift=shift,
            timesteps=timesteps,
            cover_noise_strength=cover_noise_strength,
            repaint_mask=repaint_mask,
            clean_src_latents=clean_src_latents,
            repaint_crossfade_frames=repaint_crossfade_frames,
            repaint_injection_ratio=repaint_injection_ratio,
        )
    patched.__aceflow_infer_patch__ = True
    patched.__aceflow_infer_patch_variant__ = "turbo"
    return patched


def _patch_target(module_name: str, variant: str, kind: str, supports_timesteps: bool) -> tuple[bool, str]:
    try:
        module = _import_module(module_name)
    except Exception as exc:
        return False, f"module_import_failed:{exc.__class__.__name__}"

    cls = getattr(module, "AceStepConditionGenerationModel", None)
    if cls is None:
        return False, "class_not_found"

    original_fn = getattr(cls, "generate_audio", None)
    if original_fn is None:
        return False, "method_not_found"
    if getattr(original_fn, "__aceflow_infer_patch__", False):
        return True, "already_patched"

    try:
        if kind == "quality":
            patched_fn = _make_quality_patched_fn(original_fn, variant=variant, module_globals=module.__dict__, supports_timesteps=supports_timesteps)
        elif kind == "turbo":
            patched_fn = _make_turbo_patched_fn(original_fn, module_globals=module.__dict__)
        else:
            return False, "unsupported_patch_kind"
        setattr(cls, "generate_audio", patched_fn)
    except Exception as exc:
        return False, f"patch_install_failed:{exc.__class__.__name__}"

    return True, "patched"


def _resolve_runtime_patch_plan(cls: type, original_fn: Callable[..., Any]) -> tuple[str, str, bool]:
    module_name = str(getattr(cls, "__module__", "") or "")
    lowered = module_name.lower()
    try:
        params = inspect.signature(original_fn).parameters
    except Exception:
        params = {}

    if "turbo" in lowered and "sft" not in lowered:
        return "turbo", "turbo", True
    if ".sft." in lowered or lowered.endswith(".sft") or "modeling_acestep_v15_sft" in lowered:
        return "sft", "quality", True
    if ".base." in lowered or lowered.endswith(".base") or "modeling_acestep_v15_base" in lowered:
        return "base", "quality", "timesteps" in params

    if "fix_nfe" in params:
        return "turbo", "turbo", True
    if "infer_steps" in params:
        return ("sft" if "timesteps" in params else "base"), "quality", ("timesteps" in params)
    raise RuntimeError(f"unsupported_runtime_model:{module_name or cls!r}")


def _patch_runtime_model(model: Any) -> tuple[bool, dict]:
    cls = getattr(model, "__class__", None)
    target_name = f"{getattr(cls, '__module__', 'unknown')}.{getattr(cls, '__name__', 'Unknown')}"
    if cls is None:
        return False, {"target": target_name, "variant": "unknown", "kind": "unknown", "status": "class_missing"}

    original_fn = getattr(cls, "generate_audio", None)
    if original_fn is None:
        return False, {"target": target_name, "variant": "unknown", "kind": "unknown", "status": "method_not_found"}
    if getattr(original_fn, "__aceflow_infer_patch__", False):
        variant = str(getattr(original_fn, "__aceflow_infer_patch_variant__", "runtime") or "runtime")
        kind = "turbo" if variant == "turbo" else "quality"
        return True, {"target": target_name, "variant": variant, "kind": kind, "status": "already_patched"}

    module_name = str(getattr(cls, "__module__", "") or "")
    try:
        module = sys.modules.get(module_name) or importlib.import_module(module_name)
        variant, kind, supports_timesteps = _resolve_runtime_patch_plan(cls, original_fn)
        if kind == "turbo":
            patched_fn = _make_turbo_patched_fn(original_fn, module_globals=module.__dict__)
        else:
            patched_fn = _make_quality_patched_fn(
                original_fn,
                variant=variant,
                module_globals=module.__dict__,
                supports_timesteps=supports_timesteps,
            )
        setattr(cls, "generate_audio", patched_fn)
        status = "patched"
        ok = True
    except Exception as exc:
        variant = "unknown"
        kind = "unknown"
        status = f"patch_failed:{exc.__class__.__name__}:{exc}"
        ok = False

    return ok, {"target": target_name, "variant": variant, "kind": kind, "status": status}


def _patch_service_generate_execute_target() -> tuple[bool, str, Callable[..., Any] | None]:
    module_name = "acestep.core.generation.handler.service_generate_execute"
    qualname = f"{module_name}.ServiceGenerateExecuteMixin._execute_service_generate_diffusion"
    try:
        module = sys.modules.get(module_name) or importlib.import_module(module_name)
    except Exception as exc:
        return False, f"module_import_failed:{exc.__class__.__name__}", None

    cls = getattr(module, "ServiceGenerateExecuteMixin", None)
    if cls is None:
        return False, "class_not_found", None

    original_fn = getattr(cls, "_execute_service_generate_diffusion", None)
    if original_fn is None:
        return False, "method_not_found", None
    if getattr(original_fn, "__aceflow_service_infer_patch__", False):
        return True, "already_patched", original_fn

    @wraps(original_fn)
    def patched(self, payload, generate_kwargs, seed_param, infer_method, shift, audio_cover_strength):
        model = getattr(self, "model", None)
        if model is not None:
            ok, info = _patch_runtime_model(model)
            last_target = getattr(self, "_aceflow_infer_runtime_target", None)
            current_target = (info.get("target"), info.get("status"))
            if last_target != current_target:
                _emit_runtime_line(
                    f"{_PATCH_LOG_PREFIX} runtime_target={info.get('target')} variant={info.get('variant')} kind={info.get('kind')} ok={bool(ok)} status={info.get('status')}"
                )
                try:
                    setattr(self, "_aceflow_infer_runtime_target", current_target)
                except Exception:
                    pass
        return original_fn(self, payload, generate_kwargs, seed_param, infer_method, shift, audio_cover_strength)

    patched.__aceflow_service_infer_patch__ = True
    setattr(cls, "_execute_service_generate_diffusion", patched)
    return True, "patched", patched


def install_runtime_infer_method_patch() -> dict:
    target_name = "acestep.core.generation.handler.service_generate_execute.ServiceGenerateExecuteMixin._execute_service_generate_diffusion"
    ok, status, _ = _patch_service_generate_execute_target()
    results = [{
        "module": target_name,
        "variant": "runtime",
        "kind": "dispatcher",
        "ok": bool(ok),
        "status": status,
    }]
    summary = {
        "installed": bool(ok),
        "patched_modules": [target_name] if ok else [],
        "results": results,
        "supported_methods": list(VALID_INFER_METHODS),
    }
    state = "ready" if ok else "partial"
    _emit_runtime_line(
        f"{_PATCH_LOG_PREFIX} install status={state} patched={1 if ok else 0}/1 methods={','.join(VALID_INFER_METHODS)}"
    )
    _emit_runtime_line(
        f"{_PATCH_LOG_PREFIX} target={target_name} variant=runtime kind=dispatcher ok={bool(ok)} status={status}"
    )
    return summary
