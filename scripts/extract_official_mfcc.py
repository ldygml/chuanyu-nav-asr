import os, sys
import torch, torchaudio, kaldiio, numpy as np

wav_scp = "/root/asr-competition/official_data/wav.scp"
feat_dir = "/root/asr-competition/official_data/feat"
nj = 8
os.makedirs(feat_dir, exist_ok=True)

# Read all utterances
utts = []
with open(wav_scp) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            utts.append((parts[0], parts[1]))

print(f"Total: {len(utts)} utterances")

# Split into nj parts
chunk_size = (len(utts) + nj - 1) // nj

# MFCC transform
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=40,
    melkwargs={'n_mels': 40, 'n_fft': 512, 'win_length': 400,
               'hop_length': 160, 'f_min': 40, 'f_max': 7800,
               'window_fn': torch.hamming_window}
)

for n in range(nj):
    chunk = utts[n*chunk_size:(n+1)*chunk_size]
    if not chunk:
        continue

    out_dir = f"{feat_dir}/{n+1}"
    os.makedirs(out_dir, exist_ok=True)

    ark_path = f"{out_dir}/mfcc.{n+1}.ark"
    scp_path = f"{out_dir}/mfcc.{n+1}.scp"
    len_path = f"{out_dir}/feat2len.{n+1}.txt"

    feat_dict = {}
    feat_lens = []

    for utt_id, wav_path in chunk:
        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform[0:1]

            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1).numpy()
            feat_dict[utt_id] = mfcc.astype(np.float32)
            feat_lens.append(f"{utt_id} {mfcc.shape[0]}")
        except Exception as e:
            print(f"Error {utt_id}: {e}")

    if feat_dict:
        kaldiio.save_ark(ark_path, feat_dict, scp=scp_path)
        with open(len_path, 'w') as f:
            f.write("\n".join(feat_lens) + "\n")
        print(f"Part {n+1}: {len(feat_dict)} saved")

# Merge
with open(f"{feat_dir}/mfcc.scp", 'w') as out:
    for n in range(nj):
        scp = f"{feat_dir}/{n+1}/mfcc.{n+1}.scp"
        if os.path.exists(scp):
            with open(scp) as f:
                out.write(f.read())
with open(f"{feat_dir}/feat2len.txt", 'w') as out:
    for n in range(nj):
        lp = f"{feat_dir}/{n+1}/feat2len.{n+1}.txt"
        if os.path.exists(lp):
            with open(lp) as f:
                out.write(f.read())

print("Done!")
