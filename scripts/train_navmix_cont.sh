#!/bin/bash
# ============================================================
# 从 navmix checkpoint_best.pt 继续训练 40k 步
# 预计: ~16h，明天中午前完成
# 保存到: /Save/checkpoints_nav_mix_cont/
# ============================================================
set -euo pipefail

export PYTHONPATH=/root/asr-competition/submission/fairseq:$PYTHONPATH
cd /root/asr-competition/submission/fairseq

echo "[train] 启动继续训练..."
echo "[train] 从: /Save/checkpoints_nav_mix/checkpoint_best.pt"
echo "[train] 保存到: /Save/checkpoints_nav_mix_cont"
echo "[train] max_update=40000, lr=5e-06"

/opt/conda/envs/chuanyu-ASR/bin/python3 fairseq_cli/train.py \
    common.user_dir=/root/asr-competition/ASR/data2vec_dialect \
    common.fp16=false \
    common.seed=777 \
    common.log_format=json \
    common.log_interval=200 \
    common.tensorboard_logdir=tb \
    model.w2v_path=/root/asr-competition/model_checkpoints/kespeech_encoder.pt \
    model.freeze_finetune_updates=0 \
    task.data=/root/asr-competition/train_data \
    +task.target_dictionary=/root/asr-competition/model_checkpoints \
    checkpoint.save_dir=/Save/checkpoints_nav_mix_cont \
    checkpoint.restore_file=/Save/checkpoints_nav_mix/checkpoint_best.pt \
    checkpoint.reset_lr_scheduler=true \
    checkpoint.reset_optimizer=true \
    checkpoint.keep_best_checkpoints=-1 \
    checkpoint.no_epoch_checkpoints=true \
    checkpoint.best_checkpoint_metric=uer \
    dataset.max_tokens=50000 \
    dataset.max_tokens_valid=50000 \
    dataset.num_workers=6 \
    optimization.max_update=40000 \
    optimization.update_freq='[1]' \
    optimization.lr='[5e-06]' \
    optimization.sentence_avg=true \
    lr_scheduler.max_update=40000 \
    optimizer._name=adam \
    'optimizer.adam_betas=(0.9,0.98)' \
    optimizer.adam_eps=1e-08 \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true \
    distributed_training.distributed_world_size=1 \
    2>&1 | tee /Save/checkpoints_nav_mix_cont/train.log

echo "[train] 完成"
