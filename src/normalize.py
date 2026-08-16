"""
文本规范化脚本：支持两种输入格式的文本规范化处理。

    模式1 - tsv : 从 UTTERANCEINFO TSV 文件中提取 wav 文件 ID 和标注文本
    模式2 - jsonl: 对模型转录结果 JSONL 进行文本规范化（去标点）

用法:
    python normalize.py                        # 默认 TSV 模式
    python normalize.py -m jsonl -i hyp.jsonl  # 模型转录结果模式
    python normalize.py -m tsv -i input.txt -o output.jsonl
"""

import argparse
import json
import re


def clean_text(raw: str) -> str:
    """去除标点符号和空白，只保留中文字符、英文字母和数字。"""
    return re.sub(r"[^一-鿿A-Za-z0-9]", "", raw)


# ---- 处理函数：UTTERANCEINFO TSV 原始文件 -----------------------------------

def normalize_tsv(input_path: str, output_path: str) -> int:
    """读取 TSV 文件，提取 id 和 text 并写入 JSONL。

    TSV 格式（无表头的 6 列）:
        CHANNEL  UTTRANS_ID  SPEAKER_ID  PROMPT  PROMTTYPE  TRANSCRIPTION

    处理逻辑:
        1. 取第 2 列（UTTRANS_ID）去掉 .wav 后缀作为 id
        2. 取第 6 列（TRANSCRIPTION）去除标点/空白，保留中文、英文、数字作为 text
        3. 跳过第一行及任何不含合法 .wav 文件名的行

    Returns:
        成功写入的记录数。
    """
    count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            # TSV 按制表符拆分为 6 列
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            wav_filename = parts[1]
            if not wav_filename.endswith(".wav"):
                continue

            # 去掉 .wav 后缀得到 id
            uid = wav_filename[:-4]
            raw_text = parts[5]

            item = {"id": uid, "text": clean_text(raw_text)}
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    return count


# ---- 处理函数：模型转录结果 JSONL -------------------------------------------

def normalize_jsonl(input_path: str, output_path: str) -> int:
    """读取模型转录结果 JSONL，对 text 字段做规范化后重新输出。

    输入 JSONL 格式:
        {"id": "...", "text": "..."}

    处理逻辑:
        1. 保留 id 不变
        2. 对 text 去除标点/空白，保留中文、英文、数字

    Returns:
        成功写入的记录数。
    """
    count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            item = json.loads(line.strip())
            item["text"] = clean_text(item["text"])
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    return count


# ---- 入口 ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="文本规范化：支持 TSV 原始文件和 JSONL 模型转录结果"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["tsv", "jsonl"],
        default="tsv",
        help="输入文件模式: tsv=原始标注文件, jsonl=模型转录结果 (默认: tsv)",
    )
    parser.add_argument(
        "-i", "--input",
        help="输入文件路径（TSV 模式默认: UTTERANCEINFO.txt）",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 JSONL 文件路径（默认: utterance.jsonl）",
    )
    args = parser.parse_args()

    # 根据模式设置默认路径
    if args.mode == "tsv":
        input_path = args.input or "UTTERANCEINFO.txt"
        output_path = args.output or "utterance.jsonl"
        n = normalize_tsv(input_path, output_path)
    else:
        if not args.input:
            parser.error("JSONL 模式必须指定 -i/--input")
        input_path = args.input
        output_path = args.output or "utterance.jsonl"
        n = normalize_jsonl(input_path, output_path)

    print(f"完成。共写入 {n} 条记录到 {output_path}")


if __name__ == "__main__":
    main()
