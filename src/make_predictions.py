#!/usr/bin/env python3
"""
从 infer.log 提取 HYPO 行，按数据集索引还原顺序后写入 predictions.jsonl。

关键：fairseq 按音频长度重排 batch，HYPO 输出顺序 ≠ manifest 顺序。
     infer.py 在每条 HYPO 后附加 IDX=N（数据集索引），
     本脚本解析 IDX 并按索引排序，确保 ID 与文本正确对齐。
"""
import argparse
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_log", required=True, help="infer.py 输出日志")
    parser.add_argument("--input_manifest", required=True, help="原始 manifest.jsonl")
    parser.add_argument("--output_path", required=True, help="predictions.jsonl 输出路径")
    args = parser.parse_args()

    # 读取 manifest 获取 id 列表（保持顺序）
    manifest_ids = []
    with open(args.input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            manifest_ids.append(obj["id"])

    # 从 infer.log 提取 HYPO 行及索引
    idx_pattern = re.compile(r"IDX=(\d+)")
    id_hypos = []  # [(idx, text), ...]

    with open(args.infer_log, "r", encoding="utf-8") as f:
        for line in f:
            if "HYPO:" in line:
                # 格式: ... HYPO: 转录文本\tIDX=N
                text = line.split("HYPO:", 1)[1].strip().replace("<unk>", "")
                match = idx_pattern.search(text)
                if match:
                    idx = int(match.group(1))
                    text = idx_pattern.sub("", text).rstrip()
                    id_hypos.append((idx, text))

    # 按索引排序，还原原始顺序
    id_hypos.sort(key=lambda x: x[0])

    if len(id_hypos) != len(manifest_ids):
        print(
            "[WARN] HYPO 数量 ({}) != manifest ID 数量 ({})".format(
                len(id_hypos), len(manifest_ids)
            )
        )

    # 写入 predictions.jsonl（按 manifest 顺序）
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    count = 0
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, uid in enumerate(manifest_ids):
            text = id_hypos[i][1] if i < len(id_hypos) else ""
            f.write(json.dumps({"id": uid, "text": text}, ensure_ascii=False) + "\n")
            count += 1

    print("[make_predictions] {} 条 -> {}".format(count, args.output_path))


if __name__ == "__main__":
    main()
