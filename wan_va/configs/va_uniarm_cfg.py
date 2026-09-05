# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""UniArmL1 config for pick_key_and_controller_335 (6D: 5 joints + gripper)."""
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_uniarm_cfg = EasyDict(__name__='Config: VA uniarm')
va_uniarm_cfg.update(va_shared_cfg)
va_uniarm_cfg.infer_mode = 'server'
# 单卡 24GB：VAE + UMT5 放到 CPU，只把 transformer 留在 GPU（官方 README 推荐）
va_uniarm_cfg.enable_offload = True

va_uniarm_cfg.wan22_pretrained_model_name_or_path = (
    '/home/karthus_chen/ycb_ws/model/lingbot-va-base')

# Optional: used when client does not send negative_prompt_embeds.
va_uniarm_cfg.empty_emb_path = (
    '/home/karthus_chen/unitree_sh_disk/tools/ycb/datasets/uniarml1/'
    'pick_key_and_controller_335/lerobot_v2.1/empty_emb.pt'
)
va_uniarm_cfg.attn_window = 30
va_uniarm_cfg.frame_chunk_size = 4
va_uniarm_cfg.env_type = 'none'

va_uniarm_cfg.height = 256
va_uniarm_cfg.width = 256
va_uniarm_cfg.action_dim = 30
# Existing latents: 10fps from 30Hz (stride=3) -> action_per_frame = 3*4 = 12
va_uniarm_cfg.action_per_frame = 12
# Training attention backend. Official recipe uses "flex"; "flashattn" is faster
# on this PPU but drops FlexAttention causal/window masks.
va_uniarm_cfg.train_attn_mode = "flashattn"
# Truncate each train sample along latent time to fit 96GB.
va_uniarm_cfg.train_max_latent_frames = 8
# Cap flex attention window (only used when train_attn_mode=flex).
va_uniarm_cfg.train_window_size_range = (4, 9)
va_uniarm_cfg.obs_cam_keys = [
    'observation.images.head',
    'observation.images.wrist',
]
va_uniarm_cfg.guidance_scale = 5
va_uniarm_cfg.action_guidance_scale = 1

va_uniarm_cfg.num_inference_steps = 5
va_uniarm_cfg.video_exec_step = -1
va_uniarm_cfg.action_num_inference_steps = 10

va_uniarm_cfg.snr_shift = 5.0
va_uniarm_cfg.action_snr_shift = 1.0

# 6D action -> slots 0-4 (joints as left-EEF proxy) + 28 (left gripper)
va_uniarm_cfg.used_action_channel_ids = list(range(0, 5)) + [28]
inverse_used_action_channel_ids = (
    [len(va_uniarm_cfg.used_action_channel_ids)] * va_uniarm_cfg.action_dim)
for i, j in enumerate(va_uniarm_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_uniarm_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_uniarm_cfg.action_norm_method = 'quantiles'
# From meta/stats.json action q01/q99, mapped into 30D slots 0-4 and 28.
va_uniarm_cfg.norm_stat = {
    'q01': [
        -0.4068098444218262,
        -0.754788718291197,
        -1.6265019416814068,
        -1.5423028696515446,
        -0.3441322742806354,
    ] + [0.0] * 23 + [0.0020311450030420005, 0.0],
    'q99': [
        1.0995928362434875,
        1.8618439359244199,
        0.8450650801155486,
        0.3464779345416733,
        1.3929818492672859,
    ] + [0.0] * 23 + [0.0282233907722573, 0.0],
}
