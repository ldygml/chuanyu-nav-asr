script = r'''. ./path.sh || exit 1

pretrained_model=/root/asr-competition/model_checkpoints/kespeech_encoder.pt

python /root/asr-competition/fairseq/fairseq_cli/hydra_train.py -m --config-dir config/v2_dialect_asr \
    --config-name base_audio_finetune_140h \
    common.user_dir=/root/asr-competition/ASR/data2vec_dialect \
    common.fp16=false \
    model.w2v_path=${pretrained_model} \
    model.freeze_finetune_updates=15000 \
    model.activation_dropout=0.2 \
    model.final_dropout=0.1 \
    task.data=/root/asr-competition/train_data \
    +task.target_dictionary=/root/asr-competition/model_checkpoints \
    checkpoint.save_dir=/Save/checkpoints_v7 \
    checkpoint.keep_best_checkpoints=-1 \
    dataset.max_tokens=20000 \
    dataset.max_tokens_valid=20000 \
    optimization.update_freq='[4]' \
    optimization.lr='[5e-06]'
'''
with open('/root/asr-competition/ASR/data2vec_dialect/run_scripts/run_d2v_finetune.sh', 'w') as f:
    f.write(script)
print('done')
