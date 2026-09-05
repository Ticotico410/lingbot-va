# UniArmL1 微调指南（`pick_key_and_controller_335`）

数据根目录：

```
/mnt/workspace/users/wanganran_2T/datasets/uniarml1/pick_key_and_controller_335/lerobot_v2.1
```

现状：LeRobot **v2.1**，`uniarml1`，**action/state=6D**，相机 `head`+`wrist`，30fps，334 ep。缺 `action_config`、`latents/`、`empty_emb.pt`。

标准 30D 槽位：`0–6` 左EEF / `7–13` 右EEF / `14–20` 左关节 / `21–27` 右关节 / `28` 左爪 / `29` 右爪。未用维填 0。

---

## 0. 环境

```bash
cd /home/karthus_chen/ycb_ws/lingbot-va
# PyTorch 2.9 + CUDA 12.6（见 README）
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
# 关键：lerobot==0.3.3、diffusers==0.36.0、transformers==4.55.2、easydict、wandb、scipy

source /mnt/workspace/users/wanganran_2T/lingbot-va/.venv/bin/activate
```

---

## 1. 下载权重

```bash
# 后训练起点（含 transformer / vae / text_encoder / tokenizer）
huggingface-cli download robbyant/lingbot-va-base --local-dir /mnt/workspace/users/wanganran_2T/ckpt/
```

训：只用其中的 `transformer/`。提 latent / 文本：用同包里的 `vae/` + `text_encoder/` + `tokenizer/`。

训练前把 `<ckpt>/transformer/config.json` 的 `attn_mode` 设为 `"flex"`；推理改回 `"torch"` / `"flashattn"`。

---

## 2. 数据准备（只改数据，不改训练代码）

### 2.1 `meta/episodes.jsonl` 加 `action_config`

每行补：

```json
"action_config": [{"start_frame": 0, "end_frame": <length>, "action_text": "<tasks 文本>"}]
```

### 2.2 抽 Wan2.2 VAE latent + text_emb

- resize ≈ **256×256**，目标采样 **~7.5–15 fps**（建议 10fps；`action_per_frame = (ori_fps/target_fps)*4 = 12`）
- 输出放到 `latents/chunk-XXX/<相机键>/episode_{idx:06d}_{start}_{end}.pth`
- 字段见 README（`latent`/`text_emb`/`frame_ids`/`fps`/`ori_fps` 等）
- 相机键必须与 cfg 一致：`observation.images.head`、`observation.images.wrist`

```bash
source .venv/bin/activate
# 单卡全量
python script/extract_latents.py \
  --dataset_path /mnt/workspace/users/wanganran_2T/datasets/uniarml1/pick_key_and_controller_335/lerobot_v2.1 \
  --ckpt_path /mnt/workspace/users/wanganran_2T/ckpt/lingbot-va-base \
  --target_fps 10 --height 256 --width 256 --device cuda:0

# 或双卡分片（示例）
python script/extract_latents.py ... --start_ep 0 --end_ep 167 --device cuda:0
python script/extract_latents.py ... --start_ep 167 --end_ep 334 --device cuda:1
```

### 2.3 生成 `empty_emb.pt`

`extract_latents.py` 会一并写出 `dataset_path/empty_emb.pt`（UMT5 编码空串，shape `[512, 4096]`）。也可：

```bash
python script/extract_latents.py --dataset_path ... --ckpt_path ... --empty_emb_only
```

---

## 3. 新本体配置（核心改动）

仿 `va_demo_cfg.py` + `va_demo_train_cfg.py` 新建：

| 文件 | 作用 |
|---|---|
| `wan_va/configs/va_uniarm_cfg.py` | 本体/相机/动作槽/归一化 |
| `wan_va/configs/va_uniarm_train_cfg.py` | 数据路径与训练超参 |
| `wan_va/configs/__init__.py` | 注册 `'uniarm_train': va_uniarm_train_cfg` |

### 3.1 `va_uniarm_cfg.py`（相对 demo 改这些）

| 字段 | 建议值 | 说明 |
|---|---|---|
| `wan22_pretrained_model_name_or_path` | `/mnt/workspace/users/wanganran_2T/ckpt/lingbot-va-base` | 已落盘路径 |
| `env_type` | `'none'` | **不要** `robotwin_tshape`（否则会相对位姿+特殊拼图） |
| `obs_cam_keys` | `['observation.images.head','observation.images.wrist']` | 与数据目录名一致 |
| `height` / `width` | `256` / `256` | 与抽 latent 分辨率一致 |
| `action_dim` | `30` | 固定，勿改 |
| `used_action_channel_ids` | `list(range(0,5))+[28]` | 假定 6D=`5关节+爪`→槽 `0–4`+`28`；若 6 维全是关节则改为 `list(range(0,6))` 或 `list(range(14,20))` |
| `inverse_used_action_channel_ids` | 按 demo/libero 同样循环生成 | **必须**与 `used_action_channel_ids` 同步 |
| `norm_stat` | 用数据 `meta/stats.json` 的 action `q01/q99` 填进对应槽，其余 `0`，爪 `q99` 可留真实值 | 长度均为 30 |
| `action_per_frame` | `12`（10fps）或 `8`（15fps）或 `16`（7.5fps） | `= (30/目标fps)*4`，须与 latent 的 `frame_ids` 步长一致 |
| `frame_chunk_size` / `attn_window` | `4` / `30` | 可先跟 demo |
| `action_snr_shift` | `1.0` | 可先跟 demo；难收敛再试更小 |

本数据 `stats.json` 可直接用的 q01/q99（映射到槽 0–4 与 28）：

```
q01: [-0.407, -0.755, -1.627, -1.542, -0.344] + [0]*23 + [0.002, 0]
q99: [ 1.100,  1.862,  0.845,  0.346,  1.393] + [0]*23 + [0.028, 0]
```

（若改槽位映射，上述数要挪到对应下标。）

### 3.2 `va_uniarm_train_cfg.py`

```python
# 已写入 wan_va/configs/va_uniarm_train_cfg.py
dataset_path = '.../pick_key_and_controller_335/lerobot_v2.1'
empty_emb_path = os.path.join(dataset_path, 'empty_emb.pt')
learning_rate = 1e-5
batch_size = 1
gradient_accumulation_steps = 8   # 有效 batch≈ NGPU*8
num_steps = 20000
enable_wandb = False              # 需要时再开，并填 run_va_posttrain.sh 的 WANDB_*
```

启动：

```bash
source .venv/bin/activate
# transformer/config.json 的 attn_mode 需为 "flex"（当前 base 已是 flex）
NGPU=2 CONFIG_NAME='uniarm_train' bash script/run_va_posttrain.sh
```

### 3.3 一般不用改代码

| 文件 | 何时才改 |
|---|---|
| `wan_va/dataset/lerobot_latent_dataset.py` | 仅当动作需要相对位姿等特殊变换（`env_type=='robotwin_tshape'` 分支）；单臂绝对关节/EEF **不用改** |
| `wan_va/train.py` | 一般不用 |
| `wan_va/modules/model.py` | `action_dim=30` 已固定，不要动 |

动作对齐逻辑（已有）：原始 `action[T,6]` → pad 一列 0 → 按 `inverse_used_action_channel_ids` 散到 30D → 分位数归一化。因此 **`len(used_action_channel_ids)` 必须 = 原始 action 维数（6）**。

---

## 4. 启动训练

入口是仓库根目录的 `train.sh`（管缓存目录 / HF / wandb / 超参），内部再调 `script/run_va_posttrain.sh`。

```bash
# tmux
tmux new -s train_lingbot_va
cd /mnt/workspace/users/wanganran_2T/lingbot-va
bash train.sh

# Detach: 
Ctrl-B then D

# Reattach:
tmux attach -t train_lingbot_va

# 结束训练进程:
pkill -KILL -f 'wan_va.train' || true
```

默认：`NUM_GPUS=2`，缓存 `/mnt/workspace/users/wanganran_2T/.cache/lingbot-va`，输出 `/mnt/workspace/users/wanganran_2T/ckpt/lingbot-va-runs/uniarm_pick_key`，`ENABLE_WANDB=0`。改路径/超参见 `train.sh` 顶部。

---

## 5. 推理侧（部署时）

再做 `va_uniarm_cfg.py` 的推理 config（或复用同一 cfg），注册到 `VA_CONFIGS`，`obs_cam_keys` / `used_action_channel_ids` / `norm_stat` / `action_per_frame` **必须与训练一致**；`wan22_pretrained_model_name_or_path` 指向微调后的 checkpoint（或 base+覆盖 transformer）。

---

## 检查清单

1. `episodes.jsonl` 有 `action_config`
2. `latents/.../episode_*_*.pth` 两个相机都齐全，且能被 `_check_meta` 找到
3. `empty_emb.pt` 在 `dataset_path`
4. `obs_cam_keys` = `head` + `wrist`
5. `env_type='none'`
6. `used_action_channel_ids` 长度=6，且与 `norm_stat` 槽位一致
7. `action_per_frame` ↔ 抽 latent 的 fps 步长
8. 权重 `lingbot-va-base`，`attn_mode=flex`
