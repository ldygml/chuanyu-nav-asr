"""
中文 CER（字错误率）计算脚本：将模型转录结果与参考文本逐条比对，
计算每条的字错误率并输出 JSONL，最后打印平均 CER。

用法:
    python score.py -r reference.jsonl -t transcription.jsonl -o result.jsonl
"""

import argparse
import json


def char_error_rate(ref: str, hyp: str) -> float:
    """计算两个字符串的字符级编辑距离并返回 CER。

    CER = (S + D + I) / N
         S: 替换数, D: 删除数, I: 插入数, N: 参考文本字符数

    若参考文本为空则返回 0.0。
    """
    if len(ref) == 0:
        return 0.0

    # 将字符串转为字符列表，以便正确处理中文字符
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n = len(ref_chars)
    m = len(hyp_chars)

    # DP 求编辑距离，仅保留当前行和前一行以节省内存
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 删除
                curr[j - 1] + 1,   # 插入
                prev[j - 1] + cost # 替换 / 正确
            )
        prev, curr = curr, prev

    return prev[m] / n


def load_reference(path: str) -> dict:
    """从 JSONL 读取参考文本，返回 {id: text} 映射。"""
    ref_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            ref_map[item["id"]] = item["text"]
    return ref_map


def compute_cer(ref_path: str, hyp_path: str, output_path: str) -> float:
    """逐条计算 CER 并输出到 JSONL，返回平均 CER。

    输出 JSONL 每行包含:
        id   - 语句编号
        text - 模型转录文本
        ref  - 参考文本
        cer  - 字错误率
    """
    ref_map = load_reference(ref_path)
    total_cer = 0.0
    count = 0

    with open(hyp_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            item = json.loads(line.strip())
            uid = item["id"]
            hyp_text = item["text"]

            if uid not in ref_map:
                print(f"警告: {uid} 在参考文件中不存在，跳过")
                continue

            ref_text = ref_map[uid]
            cer = char_error_rate(ref_text, hyp_text)

            f_out.write(
                json.dumps(
                    {"id": uid, "text": hyp_text, "ref": ref_text, "cer": round(cer, 6)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            total_cer += cer
            count += 1

    avg_cer = total_cer / count if count > 0 else 0.0
    return avg_cer, count


def main():
    parser = argparse.ArgumentParser(description="计算中文 CER（字错误率）")
    parser.add_argument(
        "-r", "--reference",
        default="utterance.jsonl",
        help="参考文本 JSONL 路径（默认: utterance.jsonl）",
    )
    parser.add_argument(
        "-t", "--transcription",
        required=True,
        help="模型转录结果 JSONL 路径",
    )
    parser.add_argument(
        "-o", "--output",
        default="cer_result.jsonl",
        help="输出结果 JSONL 路径（默认: cer_result.jsonl）",
    )
    args = parser.parse_args()

    avg_cer, count = compute_cer(args.reference, args.transcription, args.output)
    print(f"平均 CER: {avg_cer:.4f}  (共 {count} 条)")
    print(f"结果已写入: {args.output}")


if __name__ == "__main__":
    main()
