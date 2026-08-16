import sys
seed = sys.argv[1] if len(sys.argv) > 1 else '123'

script = f'''. ./path.sh || exit 1

python /root/asr-competition/fairseq/fairseq_cli/hydra_train.py -m --config-dir config/v2_dialect_asr \
    --config-name base_audio_finetune_140h \
    common.user_dir=/root/asr-competition/ASR/data2vec_dialect \
    common.fp16=false \
    common.seed={seed} \
    model.w2v_path=/root/asr-competition/model_checkpoints/kespeech_encoder.pt \
    checkpoint.finetune_from_model=/root/asr-competition/model_checkpoints/finetune_large_kespeech.pt \
    model.freeze_finetune_updates=0 \
    task.data=/root/asr-competition/train_data \
    +task.target_dictionary=/root/asr-competition/model_checkpoints \
    checkpoint.save_dir=/Save/checkpoints_seed{seed} \
    checkpoint.keep_best_checkpoints=-1 \
    dataset.max_tokens=20000 \
    dataset.max_tokens_valid=20000 \
    optimization.update_freq='[4]' \
    optimization.lr='[1e-06]'
'''

with open('/root/asr-competition/ASR/data2vec_dialect/run_scripts/run_d2v_finetune.sh', 'w') as f:
    f.write(script)

import subprocess, os
os.makedirs(f'/Save/checkpoints_seed{seed}', exist_ok=True)
subprocess.run(['bash', '-c',
    'cd /root/asr-competition/ASR/data2vec_dialect && '
    'source /opt/conda/etc/profile.d/conda.sh && conda activate chuanyu-ASR && '
    f'nohup bash run_scripts/run_d2v_finetune.sh > /root/asr-competition/finetune_output_seed{seed}.log 2>&1 &'],
    env={**os.environ, 'PYTHONUNBUFFERED': '1'})

print(f'seed{seed} started, PID check with ps aux')
