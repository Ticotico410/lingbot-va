# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_uniarm_cfg import va_uniarm_cfg

va_uniarm_train_cfg = EasyDict(__name__='Config: VA uniarm train')
va_uniarm_train_cfg.update(va_uniarm_cfg)

va_uniarm_train_cfg.dataset_path = (
    '/mnt/workspace/users/wanganran_2T/datasets/uniarml1/'
    'pick_key_and_controller_335/lerobot_v2.1')
va_uniarm_train_cfg.empty_emb_path = os.path.join(
    va_uniarm_train_cfg.dataset_path, 'empty_emb.pt')

# Disable by default; set True after exporting WANDB_* in run_va_posttrain.sh
va_uniarm_train_cfg.enable_wandb = False
# CPU DataLoader workers (needs TMPDIR on /dev/shm; see train.sh).
va_uniarm_train_cfg.load_worker = 18
va_uniarm_train_cfg.save_interval = 10000
va_uniarm_train_cfg.gc_interval = 50
va_uniarm_train_cfg.cfg_prob = 0.1

va_uniarm_train_cfg.learning_rate = 1e-5
va_uniarm_train_cfg.beta1 = 0.9
va_uniarm_train_cfg.beta2 = 0.95
va_uniarm_train_cfg.weight_decay = 1e-1
va_uniarm_train_cfg.warmup_steps = 10
va_uniarm_train_cfg.batch_size = 8
va_uniarm_train_cfg.gradient_accumulation_steps = 8
va_uniarm_train_cfg.num_steps = 100000
va_uniarm_train_cfg.save_root = './train_out/uniarm_pick_key_and_controller'
