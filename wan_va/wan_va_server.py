# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch"
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        # Used by server-mode video decode (and i2va generate()).
        self.video_processor = VideoProcessor(vae_scale_factor=1)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _streaming_vae_cache_is_cold(self) -> bool:
        return all(c is None for c in self.streaming_vae.feat_cache)

    def _encode_obs(self, obs):
        # Allow client to pass precomputed Wan VAE latents (training latents/).
        if obs.get("obs_latent") is not None:
            lat = obs["obs_latent"]
            if isinstance(lat, np.ndarray):
                lat = torch.from_numpy(np.array(lat, copy=True))
            lat = lat.to(device=self.device, dtype=self.dtype)
            if lat.ndim == 4:  # C,F,H,W
                lat = lat.unsqueeze(0)
            logger.info("Using precomputed obs_latent %s (skip VAE encode)", tuple(lat.shape))
            # Streaming VAE must still see the first frame, otherwise later KV
            # keyframe encodes (T%4==0 continuation) crash with shape mismatch.
            if obs.get("obs") is not None and self._streaming_vae_cache_is_cold():
                images = obs["obs"]
                if not isinstance(images, list):
                    images = [images]
                logger.info(
                    "Warming streaming VAE with first obs frame (cache was cold after obs_latent)"
                )
                self._encode_obs_image_list(images[:1])
            return lat

        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        # Cold cache + multi-frame is invalid for Wan streaming (needs first T=1,
        # then chunks of 4). Official clients always encode the first obs alone.
        if len(images) > 1 and self._streaming_vae_cache_is_cold():
            logger.info(
                "Streaming VAE cache cold with T=%d; encode as T=1 + groups of 4",
                len(images),
            )
            parts = [self._encode_obs_image_list(images[:1])]
            rest = images[1:]
            pad = (4 - (len(rest) % 4)) % 4
            if pad:
                rest = list(rest) + [rest[-1]] * pad
            for i in range(0, len(rest), 4):
                parts.append(self._encode_obs_image_list(rest[i:i + 4]))
            # Drop padding latent frames if we padded video frames.
            out = torch.cat(parts, dim=2)
            # Expected latent frames from video: 1 + len(rest_unpadded)//4
            n_video = len(images)
            n_latent = 1 + (n_video - 1) // 4
            return out[:, :, :n_latent]

        return self._encode_obs_image_list(images)

    def _encode_obs_image_list(self, images: list):
        """Encode a list of cam-dicts with streaming VAE → ``[1,C,F,H,W]`` on device."""
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        t0 = time.perf_counter()
        moved = False
        if self.enable_offload:
            try:
                if next(self.vae.parameters()).device.type == "cpu":
                    free_bytes, _ = torch.cuda.mem_get_info(self.device)
                    if free_bytes > 4 * (1024**3):
                        logger.info("Moving VAE -> %s for encode_obs", self.device)
                        self.vae.to(self.device)
                        moved = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("VAE GPU move skipped: %s", exc)
        try:
            if self.env_type == 'robotwin_tshape':
                videos_high = videos[0] / 255.0 * 2.0 - 1.0
                videos_left_and_right = torch.cat(videos[1:],
                                                  dim=0) / 255.0 * 2.0 - 1.0
                vae_device = next(self.streaming_vae.vae.parameters()).device
                enc_out_high = self.streaming_vae.encode_chunk(
                    videos_high.to(vae_device).to(self.dtype))
                enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                    videos_left_and_right.to(vae_device).to(self.dtype))
                enc_out = torch.cat([
                    torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                    enc_out_high
                ],
                                    dim=-2)
            else:
                videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
                vae_device = next(self.streaming_vae.vae.parameters()).device
                videos_chunk = videos.to(vae_device).to(self.dtype)
                enc_out = self.streaming_vae.encode_chunk(videos_chunk)

            mu, logvar = torch.chunk(enc_out, 2, dim=1)
            latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
            latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
            mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
            video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
            out = video_latent.to(self.device)
        finally:
            if moved:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
        logger.info(
            "VAE encode_obs done in %.2fs shape=%s (T_img=%d)",
            time.perf_counter() - t0,
            tuple(out.shape),
            len(images),
        )
        return out

    def _reset(self, prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt embeds (prefer precomputed training text_emb / empty_emb)
        self.prompt_embeds, self.negative_prompt_embeds = self._resolve_prompt_embeds(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _to_prompt_embed_tensor(self, emb) -> torch.Tensor:
        """Accept [L,D] / [1,L,D] numpy|tensor → ``[1, L, D]`` on model device/dtype."""
        if emb is None:
            raise ValueError("prompt embed is None")
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(np.array(emb, copy=True))
        elif not torch.is_tensor(emb):
            emb = torch.as_tensor(emb)
        emb = emb.detach().contiguous().clone()
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        if emb.ndim != 3:
            raise ValueError(f"prompt embed must be [L,D] or [B,L,D], got {tuple(emb.shape)}")
        return emb.to(device=self.device, dtype=self.dtype)

    def _resolve_prompt_embeds(
        self,
        prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        need_neg = self.job_config.guidance_scale > 1

        if prompt_embeds is not None:
            logger.info("Using precomputed prompt_embeds (skip UMT5 encode).")
            pos = self._to_prompt_embed_tensor(prompt_embeds)
            if negative_prompt_embeds is not None:
                neg = self._to_prompt_embed_tensor(negative_prompt_embeds)
            elif need_neg:
                neg = self._load_empty_emb_tensor()
            else:
                neg = None
            return pos, neg

        if prompt is None:
            return None, None

        # Optional: resolve from job_config.prompt_emb_path / empty_emb_path
        prompt_emb_path = getattr(self.job_config, "prompt_emb_path", None)
        if prompt_emb_path and os.path.isfile(prompt_emb_path):
            logger.info("Loading prompt_embeds from %s", prompt_emb_path)
            cached = torch.load(prompt_emb_path, map_location="cpu", weights_only=False)
            if isinstance(cached, dict):
                cached = cached.get("text_emb", cached.get("prompt_embeds"))
            pos = self._to_prompt_embed_tensor(cached)
            neg = self._load_empty_emb_tensor() if need_neg else None
            return pos, neg

        logger.info(
            "Encoding prompt with UMT5 (slow when enable_offload puts text_encoder on CPU)..."
        )
        t0 = time.perf_counter()
        pos, neg = self.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=need_neg,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info("UMT5 prompt encode done in %.1fs", time.perf_counter() - t0)
        return pos, neg

    def _load_empty_emb_tensor(self) -> torch.Tensor:
        empty_path = getattr(self.job_config, "empty_emb_path", None)
        if not empty_path or not os.path.isfile(empty_path):
            # Fall back to live encode of "" (same as training empty_emb).
            logger.warning(
                "empty_emb_path missing (%s); encoding empty string with UMT5",
                empty_path,
            )
            neg, _ = self.encode_prompt(
                prompt="",
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )
            return neg
        logger.info("Loading negative/empty prompt embeds from %s", empty_path)
        emb = torch.load(empty_path, map_location="cpu", weights_only=False)
        if isinstance(emb, dict):
            emb = emb.get("text_emb", emb.get("empty_emb", emb))
        return self._to_prompt_embed_tensor(emb)

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            n_vid = len(timesteps)
            n_act = len(action_timesteps)
            logger.info(
                "Denoise start: video_steps=%d action_steps=%d cfg=%s "
                "(first chunk is slow; progress logged each step)",
                n_vid,
                n_act,
                self.use_cfg,
            )
            for i, t in enumerate(tqdm(timesteps)):
                step_t0 = time.perf_counter()
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                logger.info(
                    "video denoise %d/%d done in %.2fs",
                    i + 1,
                    n_vid,
                    time.perf_counter() - step_t0,
                )

            for i, t in enumerate(tqdm(action_timesteps)):
                step_t0 = time.perf_counter()
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]
                logger.info(
                    "action denoise %d/%d done in %.2fs",
                    i + 1,
                    n_act,
                    time.perf_counter() - step_t0,
                )

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)
        return_video = bool(obs.get('return_video', False))

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(
                prompt=prompt,
                prompt_embeds=obs.get("prompt_embeds", None),
                negative_prompt_embeds=obs.get("negative_prompt_embeds", None),
            )
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, latents = self._infer(obs, frame_st_id=self.frame_st_id)
            out = dict(action=action)
            if return_video:
                t0 = time.perf_counter()
                out["video"] = self._latents_to_cam_videos(latents)
                logger.info(
                    "Decoded pred video for client in %.1fs (cams=%s frames=%s)",
                    time.perf_counter() - t0,
                    list(out["video"].keys()),
                    {k: v.shape for k, v in out["video"].items()},
                )
            return out

    def _latents_to_cam_videos(self, latents: torch.Tensor) -> dict:
        """Decode chunk latents → per-camera uint8 clips ``{cam_short: (F,H,W,3)}``."""
        n_cam = len(self.job_config.obs_cam_keys)
        moved = False
        vae_device = next(self.vae.parameters()).device
        if self.enable_offload and vae_device.type == "cpu":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                # Wan2.2 VAE ~1–2GB; keep a margin for decode activations.
                if free_bytes > 4 * (1024**3):
                    logger.info("Temporarily moving VAE to %s for video decode", self.device)
                    self.vae.to(self.device)
                    moved = True
                    vae_device = self.device
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skip GPU VAE move for decode: %s", exc)

        try:
            video = self.decode_one_video(latents.to(vae_device), "np")
            # Diffusers postprocess: (B, F, H, W, C) in [0, 1] float or uint8.
            frames = video[0]
            if frames.dtype != np.uint8:
                frames = (np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
            frames = np.ascontiguousarray(frames)

            out: dict = {}
            if self.env_type == "robotwin_tshape":
                # Latents are stacked along height: wrists on top, high cam below.
                # Approximate split: last third is high, first two-thirds wrists.
                f, h, w, c = frames.shape
                h_wrist = (h * 2) // 3
                wrist = frames[:, :h_wrist]
                high = frames[:, h_wrist:]
                # Split wrist strip into left/right if >=2 wrist cams.
                wrist_cams = self.job_config.obs_cam_keys[1:]
                if len(wrist_cams) >= 2:
                    w_each = w // len(wrist_cams)
                    for i, key in enumerate(wrist_cams):
                        short = key.replace("observation.images.", "")
                        out[short] = np.ascontiguousarray(
                            wrist[:, :, i * w_each:(i + 1) * w_each]
                        )
                elif len(wrist_cams) == 1:
                    short = wrist_cams[0].replace("observation.images.", "")
                    out[short] = np.ascontiguousarray(wrist)
                high_key = self.job_config.obs_cam_keys[0].replace(
                    "observation.images.", ""
                )
                out[high_key] = np.ascontiguousarray(high)
            else:
                # env_type none / default: cameras concatenated along width.
                f, h, w, c = frames.shape
                if w % n_cam != 0:
                    logger.warning(
                        "Pred video width %s not divisible by n_cam=%s; "
                        "returning single wide clip as 'pred'",
                        w,
                        n_cam,
                    )
                    out["pred"] = frames
                else:
                    w_each = w // n_cam
                    for i, key in enumerate(self.job_config.obs_cam_keys):
                        short = key.replace("observation.images.", "")
                        out[short] = np.ascontiguousarray(
                            frames[:, :, i * w_each:(i + 1) * w_each]
                        )
            return out
        finally:
            if moved:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
