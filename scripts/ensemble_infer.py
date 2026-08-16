import sys, torch, editdistance, re

sys.path.insert(0, '/root/asr-competition/fairseq')
sys.path.insert(0, '/root/asr-competition/ASR/data2vec_dialect')

from fairseq import tasks, checkpoint_utils

# Simple task setup
from audio_finetuning import SpecFinetuningTask
from fairseq.dataclass.configs import FairseqDataclass

import argparse, os
import numpy as np

# Setup task
task = SpecFinetuningTask.setup_task(
    data='/root/asr-competition/valid_data',
    labels='ltr',
    normalize=False,
    max_sample_size=4000,
    min_sample_size=100,
    target_dictionary='/root/asr-competition/model_checkpoints'
)

# Setup model
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtcConfig, Wav2VecCtc

model_paths = [
    '/Save/checkpoints_v8/checkpoint_best.pt',
    '/Save/checkpoints_seed123/checkpoint_best.pt',
    '/Save/checkpoints_seed456/checkpoint_best.pt',
]

# Load all 3 models
models = []
for path in model_paths:
    state = checkpoint_utils.load_checkpoint_to_cpu(path)
    model = Wav2VecCtc.build_model(Wav2VecCtcConfig(
        w2v_path='/root/asr-competition/model_checkpoints/kespeech_encoder.pt',
        data='/root/asr-competition/valid_data',
        apply_mask=False, dropout_input=0, final_dropout=0, dropout=0,
        attention_dropout=0, activation_dropout=0.1,
        mask_length=3, mask_prob=0.0, mask_selection='static', mask_other=0,
        no_mask_overlap=False, mask_channel_length=64, mask_channel_prob=0.0,
        mask_channel_selection='static', mask_channel_other=0,
        no_mask_channel_overlap=False, freeze_finetune_updates=0,
        feature_grad_mult=0.0, layerdrop=0.0, normalize=True, update_alibi=True,
        checkpoint_activations=False, offload_activations=False,
        min_params_to_wrap=int(1e8), ddp_backend='legacy_ddp',
    ), task)
    model.load_state_dict(state['model'], strict=False)
    model.cuda().eval()
    models.append(model)

# Load data
task.load_dataset('dev')
dataset = task.dataset('dev')

cer_err, total_chars, ser_err = 0, 0, 0
count = 0

for idx in range(len(dataset)):
    sample = dataset[idx]
    feats = sample['source'].unsqueeze(0).cuda()  # (1, T, 40)
    padding_mask = torch.zeros(1, feats.shape[1], dtype=torch.bool).cuda()

    # Average logits from 3 models
    logits = None
    with torch.no_grad():
        for m in models:
            out = m(source=feats, padding_mask=padding_mask)
            l = out['encoder_out'].transpose(0, 1)  # (1, T', 768)

            # Get CTC output
            if hasattr(m, 'proj'):
                l = m.proj(l)

            if logits is None:
                logits = l
            else:
                logits += l
    logits = logits / len(models)  # (1, T', 7531)

    # Decode
    pred_ids = logits[0].argmax(-1)
    unique = []
    for i, h in enumerate(pred_ids):
        if i == 0 or h != pred_ids[i-1]:
            unique.append(h.item())
    hypo_ids = [h for h in unique if h != 0]
    hypo_text = task.target_dictionary.string(torch.tensor(hypo_ids))

    # Ref
    ref_tokens = sample.get('target_label', sample.get('target', None))
    if ref_tokens is not None:
        ref_tokens = ref_tokens[ref_tokens != task.target_dictionary.pad()]
    ref_text = task.target_dictionary.string(ref_tokens)

    ce = editdistance.eval(list(hypo_text.replace(' ', '')), list(ref_text.replace(' ', '')))
    tclen = len(ref_text.replace(' ', ''))
    cer_err += ce
    total_chars += tclen
    if hypo_text != ref_text:
        ser_err += 1
    count += 1

    if count % 50 == 0:
        print(f'{count}/200 done')

print(f'\n=== ENSEMBLE 3 models ===')
print(f'CER: {cer_err/total_chars*100:.2f}% ({cer_err}/{total_chars})')
print(f'SER: {ser_err/count*100:.2f}% ({ser_err}/{count})')
