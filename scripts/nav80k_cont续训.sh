#!/bin/bash
# ============================================================
# nav80k_cont 续训命令（A 榜提交模型）
# 从 /Save/checkpoints_nav80k/checkpoint_best.pt 续训 20k 步
# LR=1e-06, FP32, max_tokens=50000, seed=777
# 2026-08-12 09:45 启动，当日 13:44 完成
# （此命令从 /Save/checkpoints_nav80k_cont/train.log 的 cfg dump 重建）
# ============================================================
set -e
cd /root/asr-competition/ASR/data2vec_dialect
export PYTHONPATH=/root/asr-competition/fairseq:/root/asr-competition/ASR/data2vec_dialect:$PYTHONPATH

python /root/asr-competition/fairseq/fairseq_cli/hydra_train.py \
    --config-dir config/v2_dialect_asr \
    --config-name base_audio_finetune_140h \
    common.user_dir=/root/asr-competition/ASR/data2vec_dialect \
    common.fp16=false \
    common.seed=777 \
    common.log_format=json \
    common.log_interval=200 \
    model.w2v_path=/root/asr-competition/model_checkpoints/kespeech_encoder.pt \
    model.freeze_finetune_updates=0 \
    task.data=/root/asr-competition/train_data \
    +task.target_dictionary=/root/asr-competition/model_checkpoints \
    checkpoint.save_dir=/Save/checkpoints_nav80k_cont \
    checkpoint.finetune_from_model=/Save/checkpoints_nav80k/checkpoint_best.pt \
    checkpoint.keep_best_checkpoints=-1 \
    dataset.max_tokens=50000 \
    dataset.max_tokens_valid=50000 \
    optimization.max_update=20000 \
    'optimization.update_freq=[1]' \
    'optimization.lr=[1e-06]' \
    > /Save/checkpoints_nav80k_cont/train.log 2>&1 &

# 训练配置说明
#   finetune_from_model = nav80k checkpoint_best.pt（自动重置 optimizer/lr_scheduler）
#   max_update = 20000, lr = 1e-06, update_freq = 1
#   数据 = /root/asr-competition/train_data（nav80k 同款 train/dev 划分）
#   词典 = model_checkpoints/dict.chr7531.txt（7531 字符）
#   checkpoint_best 按 dev UER 保存；save_interval = 1000
