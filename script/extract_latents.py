#!/usr/bin/env python3
"""Extract Wan2.2 VAE latents + UMT5 text_emb for LingBot-VA post-training.

Writes:
  <dataset>/latents/chunk-XXX/<cam_key>/episode_{idx:06d}_{start}_{end}.pth
  <dataset>/empty_emb.pt

Example:
  source .venv/bin/activate
  python script/extract_latents.py \\
    --dataset_path /mnt/workspace/users/wanganran_2T/datasets/uniarml1/pick_key_and_controller_335/lerobot_v2.1 \\
    --ckpt_path /mnt/workspace/users/wanganran_2T/ckpt/lingbot-va-base \\
    --target_fps 10 --height 256 --width 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wan_va"))

from modules.utils import load_text_encoder, load_tokenizer, load_vae  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--target_fps", type=float, default=10.0)
    p.add_argument("--ori_fps", type=float, default=None, help="Override; default from meta/info.json")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--cam_keys", nargs="+", default=[
        "observation.images.head",
        "observation.images.wrist",
    ])
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--chunks_size", type=int, default=None, help="Override info.json chunks_size")
    p.add_argument("--episode_indices", type=int, nargs="*", default=None,
                   help="Only process these episode indices (default: all)")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    p.add_argument("--empty_emb_only", action="store_true")
    p.add_argument("--start_ep", type=int, default=0)
    p.add_argument("--end_ep", type=int, default=None, help="Exclusive end episode index")
    return p.parse_args()


def dtype_from_str(s: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def load_episodes(meta_path: Path):
    rows = []
    with meta_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_frame_ids(start_frame: int, end_frame: int, stride: int) -> list[int]:
    """Sample [start, end) with stride, trim to Wan temporal length 1+4k."""
    ids = list(range(start_frame, end_frame, stride))
    if not ids:
        raise ValueError(f"empty frame_ids for [{start_frame}, {end_frame}) stride={stride}")
    # Wan encode: first frame + groups of 4 -> need len = 1 + 4*k
    k = (len(ids) - 1) // 4
    keep = 1 + 4 * k
    if keep < 1:
        keep = 1
    ids = ids[:keep]
    return ids


def read_video_frames(video_path: Path, frame_ids: list[int], height: int, width: int) -> torch.Tensor:
    """Return float tensor [T, H, W, 3] in [0, 255]."""
    wanted = set(frame_ids)
    max_id = max(frame_ids)
    frames = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                img = frame.to_ndarray(format="rgb24")
                frames[i] = img
            if i >= max_id and len(frames) == len(wanted):
                break
    finally:
        container.close()

    missing = [i for i in frame_ids if i not in frames]
    if missing:
        raise RuntimeError(f"{video_path}: missing frames {missing[:5]}... ({len(missing)} total)")

    arr = np.stack([frames[i] for i in frame_ids], axis=0)  # T H W C
    video = torch.from_numpy(arr).float()  # 0..255
    # resize: [T,C,H,W]
    video = video.permute(0, 3, 1, 2)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    video = video.permute(0, 2, 3, 1).contiguous()  # T H W C
    return video


def normalize_latents(latents, latents_mean, latents_std):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=torch.float32)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=torch.float32)
    return ((latents.float() - latents_mean) * latents_std).to(latents.dtype)


@torch.no_grad()
def encode_video_latent(vae, video_thwc: torch.Tensor, device, dtype) -> torch.Tensor:
    """video_thwc: [T,H,W,C] in 0..255 -> latent [f,h,w,c] bf16-ready float."""
    # [1, C, T, H, W] in [-1, 1]
    x = video_thwc.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
    x = x / 255.0 * 2.0 - 1.0
    posterior = vae.encode(x).latent_dist
    mu = posterior.mode()  # [1, z, f, h, w]
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=torch.float32)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=mu.device, dtype=torch.float32)
    mu = normalize_latents(mu, latents_mean, latents_std)
    # [f, h, w, c]
    mu = mu[0].permute(1, 2, 3, 0).contiguous()
    return mu


@torch.no_grad()
def encode_text(tokenizer, text_encoder, text: str, max_sequence_length: int, device, dtype):
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    out = text_encoder(input_ids, mask).last_hidden_state.to(dtype=dtype)
    u = out[0, : seq_lens[0]]
    pad = u.new_zeros(max_sequence_length - u.size(0), u.size(1))
    emb = torch.cat([u, pad], dim=0)  # [L, D]
    return emb.cpu()


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def latent_out_path(dataset_path: Path, chunks_size: int, cam_key: str,
                    episode_index: int, start_frame: int, end_frame: int) -> Path:
    chunk = episode_chunk(episode_index, chunks_size)
    return (
        dataset_path / "latents" / f"chunk-{chunk:03d}" / cam_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )


def save_empty_emb(tokenizer, text_encoder, dataset_path: Path, max_sequence_length, device, dtype):
    emb = encode_text(tokenizer, text_encoder, "", max_sequence_length, device, dtype)
    out = dataset_path / "empty_emb.pt"
    torch.save(emb, out)
    print(f"[ok] empty_emb.pt shape={tuple(emb.shape)} dtype={emb.dtype} -> {out}")
    return out


def process_episode(
    ep: dict,
    dataset_path: Path,
    cam_keys: list[str],
    chunks_size: int,
    stride: int,
    target_fps: float,
    ori_fps: float,
    height: int,
    width: int,
    vae,
    text_cache: dict,
    tokenizer,
    text_encoder,
    max_sequence_length: int,
    device,
    dtype,
    skip_existing: bool,
):
    episode_index = int(ep["episode_index"])
    for acfg in ep["action_config"]:
        start_frame = int(acfg["start_frame"])
        end_frame = int(acfg["end_frame"])
        action_text = acfg["action_text"]

        outs = [
            latent_out_path(dataset_path, chunks_size, cam, episode_index, start_frame, end_frame)
            for cam in cam_keys
        ]
        if skip_existing and all(p.exists() for p in outs):
            return "skip"

        frame_ids = build_frame_ids(start_frame, end_frame, stride)
        if action_text not in text_cache:
            text_cache[action_text] = encode_text(
                tokenizer, text_encoder, action_text, max_sequence_length, device, dtype
            )
        text_emb = text_cache[action_text]

        for cam_key, out_path in zip(cam_keys, outs):
            if skip_existing and out_path.exists():
                continue
            chunk = episode_chunk(episode_index, chunks_size)
            video_path = (
                dataset_path / "videos" / f"chunk-{chunk:03d}" / cam_key
                / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.exists():
                raise FileNotFoundError(video_path)

            video = read_video_frames(video_path, frame_ids, height, width)
            latent_fhwc = encode_video_latent(vae, video, device, dtype)
            latent_num_frames, latent_height, latent_width, c = latent_fhwc.shape
            latent_flat = latent_fhwc.reshape(-1, c).to(torch.bfloat16).cpu()

            payload = {
                "latent": latent_flat,
                "latent_num_frames": int(latent_num_frames),
                "latent_height": int(latent_height),
                "latent_width": int(latent_width),
                "video_num_frames": int(len(frame_ids)),
                "video_height": int(height),
                "video_width": int(width),
                "text_emb": text_emb.to(torch.bfloat16),
                "text": action_text,
                "frame_ids": frame_ids,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": int(round(target_fps)) if float(target_fps).is_integer() else target_fps,
                "ori_fps": int(round(ori_fps)) if float(ori_fps).is_integer() else ori_fps,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out_path)
        return "ok"


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    ckpt_path = Path(args.ckpt_path)
    device = torch.device(args.device)
    dtype = dtype_from_str(args.dtype)

    info = json.load(open(dataset_path / "meta" / "info.json"))
    ori_fps = float(args.ori_fps if args.ori_fps is not None else info.get("fps", 30))
    chunks_size = int(args.chunks_size if args.chunks_size is not None else info.get("chunks_size", 1000))
    stride = int(round(ori_fps / args.target_fps))
    if stride < 1:
        raise ValueError(f"invalid stride={stride} from ori_fps={ori_fps} target_fps={args.target_fps}")
    print(f"ori_fps={ori_fps} target_fps={args.target_fps} stride={stride} chunks_size={chunks_size}")
    print(f"action_per_frame hint = stride * 4 = {stride * 4}")

    print("Loading tokenizer / text_encoder / vae ...")
    tokenizer = load_tokenizer(str(ckpt_path / "tokenizer"))
    text_encoder = load_text_encoder(
        str(ckpt_path / "text_encoder"), torch_dtype=dtype, torch_device=str(device)
    )
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    save_empty_emb(tokenizer, text_encoder, dataset_path, args.max_sequence_length, device, dtype)
    if args.empty_emb_only:
        return

    vae = load_vae(str(ckpt_path / "vae"), torch_dtype=dtype, torch_device=str(device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    episodes = load_episodes(dataset_path / "meta" / "episodes.jsonl")
    if args.episode_indices is not None:
        wanted = set(args.episode_indices)
        episodes = [e for e in episodes if e["episode_index"] in wanted]
    else:
        end = args.end_ep if args.end_ep is not None else 10**9
        episodes = [e for e in episodes if args.start_ep <= e["episode_index"] < end]

    text_cache = {}
    n_ok = n_skip = n_fail = 0
    for ep in tqdm(episodes, desc="extract"):
        try:
            status = process_episode(
                ep=ep,
                dataset_path=dataset_path,
                cam_keys=args.cam_keys,
                chunks_size=chunks_size,
                stride=stride,
                target_fps=args.target_fps,
                ori_fps=ori_fps,
                height=args.height,
                width=args.width,
                vae=vae,
                text_cache=text_cache,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                max_sequence_length=args.max_sequence_length,
                device=device,
                dtype=dtype,
                skip_existing=args.skip_existing,
            )
            if status == "skip":
                n_skip += 1
            else:
                n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[fail] ep={ep.get('episode_index')}: {e}")
            raise

    print(f"done ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
