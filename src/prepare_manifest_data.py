#!/usr/bin/env python3
"""
准备推理数据：读取 manifest.jsonl，从 WAV 提取 MFCC，生成 Kaldi ARK + data.list。

输入 manifest.jsonl:
    {"id": "abc123", "audio_path": "/data/input/xxx.wav"}

输出:
    {output_dir}/mfcc.1.ark    Kaldi ARK 特征
    {output_dir}/mfcc.1.scp    ARK 索引（utt_id → ark_path:byte_offset）
    {output_dir}/data.list     fairseq 格式数据列表（7 字段 Tab 分隔）
    {output_dir}/train.tsv  →  data.list 软链接
"""
import argparse
import json
import os
import kaldiio
import torch
import torchaudio


def extract_mfcc(wav_path: str):
    """从 16kHz WAV 提取 40 维 MFCC，返回 (T, 40) float32 numpy array."""
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_mels": 40,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "f_min": 40,
            "f_max": 7800,
            "window_fn": torch.hamming_window,
        },
    )
    feats = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
    return feats.numpy().astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 manifest.jsonl
    samples = []
    with open(args.input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append((obj["id"], obj["audio_path"]))

    print(f"[prepare] 共 {len(samples)} 条音频")

    # 用 kaldiio 写入 ARK+SCP
    ark_path = os.path.join(args.output_dir, "mfcc.1.ark")
    scp_path = os.path.join(args.output_dir, "mfcc.1.scp")

    with kaldiio.WriteHelper(f"ark,scp:{ark_path},{scp_path}") as writer:
        for utt_id, wav_path in samples:
            if not os.path.exists(wav_path):
                print(f"[WARN] 音频不存在: {wav_path} (id={utt_id})")
            feats = extract_mfcc(wav_path)
            writer(utt_id, feats)

    print(f"[prepare] ARK 写入完成: {ark_path}")

    # 从 SCP 读取偏移量，生成 data.list
    data_list_path = os.path.join(args.output_dir, "data.list")
    count = 0
    with open(scp_path, "r", encoding="utf-8") as scp,          open(data_list_path, "w", encoding="utf-8") as dl:
        for line in scp:
            # SCP 格式: utt_id<TAB>ark_path:byte_offset
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            feat_ref = parts[1]  # e.g. /tmp/.../mfcc.1.ark:12345

            # 读取特征获取帧数
            feats = kaldiio.load_mat(feat_ref)
            n_frames = feats.shape[0]

            dl.write(
                f"utt:{utt_id}\tfeat:{feat_ref}\tfeat_shape:{n_frames},40\t"
                f"text:占位\ttoken:占 位\ttokenid:[0]\ttoken_shape:2,7531\n"
            )
            count += 1

    # train.tsv 软链接（fairseq 要求）
    tsv_path = os.path.join(args.output_dir, "train.tsv")
    if os.path.exists(tsv_path):
        os.remove(tsv_path)
    os.symlink("data.list", tsv_path)

    print(f"[prepare] data.list 生成完成: {count} 条")


if __name__ == "__main__":
    main()
