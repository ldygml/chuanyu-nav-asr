#!/bin/bash
# ============================================================
# 通用语音转文字工具（赛后版 pipeline）
#   任意 WAV（任意采样率/声道数，自动重采样 16k 单声道）→ 文字
# 用法:
#   bash transcribe.sh --input <wav文件 或 目录> [选项]
# 选项:
#   --input <PATH>    单个 wav 或含 wav 的目录（必填）
#   --output <FILE>   结果 jsonl 路径（默认不写文件，stdout 打印 id<TAB>text）
#   --no-hotword      跳过热词修正（默认启用）
#   --device <DEV>    推理设备，默认 cuda:0（仅支持 cuda:N）
#
# 环境变量（可选覆盖）:
#   NAV_ASR_CKPT      模型 checkpoint 路径（默认 weights/model.pt）
#   NAV_ASR_PYTHON    Python 解释器（默认取 conda chuanyu-ASR 或系统 python3）
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---- 参数解析 ----
INPUT=""; OUTPUT=""; DEVICE="cuda:0"; NO_HOTWORD=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --no-hotword) NO_HOTWORD=1; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done
if [[ -z "$INPUT" ]]; then
    echo "[ERROR] 用法: transcribe.sh --input <wav|目录> [--output out.jsonl] [--no-hotword] [--device cuda:0]"
    exit 1
fi
if [[ "$DEVICE" != cuda:* ]]; then
    echo "[ERROR] --device 仅支持 cuda:N（当前: $DEVICE）"; exit 1
fi

WORK=$(mktemp -d /tmp/transcribe_XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# ---- 收集 wav ----
if [[ -f "$INPUT" ]]; then
    WAVS=("$INPUT")
elif [[ -d "$INPUT" ]]; then
    mapfile -t WAVS < <(find "$INPUT" -maxdepth 1 -type f -iname '*.wav' | sort)
    [[ ${#WAVS[@]} -eq 0 ]] && { echo "[ERROR] 目录里没有 wav: $INPUT"; exit 1; }
else
    echo "[ERROR] 找不到输入: $INPUT"; exit 1
fi
echo "[transcribe] 共 ${#WAVS[@]} 条音频"

# ---- 生成 manifest（id = 文件名去扩展名） ----
MANIFEST="$WORK/input_manifest.jsonl"
: > "$MANIFEST"
for w in "${WAVS[@]}"; do
    id="$(basename "$w")"; id="${id%.*}"
    printf '{"id": "%s", "audio_path": "%s"}\n' "$id" "$(realpath "$w")" >> "$MANIFEST"
done

# ---- 路径解析 ----
SUBMISSION="$SCRIPT_DIR"

if [[ -n "${NAV_ASR_CKPT:-}" ]]; then
    CKPT="$NAV_ASR_CKPT"
elif [[ -f "$SCRIPT_DIR/weights/model.pt" ]]; then
    CKPT="$SCRIPT_DIR/weights/model.pt"
else
    echo "[ERROR] 找不到模型权重：请下载到 weights/model.pt 或设置 NAV_ASR_CKPT"
    exit 1
fi

if [[ -n "${NAV_ASR_PYTHON:-}" ]]; then
    PYTHON_BIN="$NAV_ASR_PYTHON"
elif [[ -x /opt/conda/envs/chuanyu-ASR/bin/python3 ]]; then
    PYTHON_BIN=/opt/conda/envs/chuanyu-ASR/bin/python3
else
    PYTHON_BIN=python3
fi

HOTWORD_FIX="$SCRIPT_DIR/hotword_fix.py"
HOTWORD_DICT="${HOTWORD_DICT:-$SCRIPT_DIR/hotword/hotword_dict_final.md}"

# 字符词典目录（含 dict.chr7531.txt）
DICT_DIR="$SCRIPT_DIR/weights"
[[ -f "$DICT_DIR/dict.chr7531.txt" ]] || DICT_DIR="$SCRIPT_DIR"

export PYTHONPATH="$SUBMISSION/fairseq:$SUBMISSION/src"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}"
echo "[transcribe] ckpt=$CKPT, python=$PYTHON_BIN"

# ---- Step 1/3: MFCC ----
echo "[transcribe] === Step 1/3: MFCC 特征 ==="
"$PYTHON_BIN" "$SUBMISSION/src/prepare_manifest_data.py" \
    --input_manifest "$MANIFEST" --output_dir "$WORK/data"

# ---- Step 2/3: ASR 推理 ----
echo "[transcribe] === Step 2/3: ASR 推理 ==="
"$PYTHON_BIN" -u "$SUBMISSION/src/infer.py" \
    --config-dir "$SUBMISSION/src/config" --config-name infer \
    task=spec_finetuning "task.data=$WORK/data" task.normalize=false \
    "common.user_dir=$SUBMISSION/src" "common_eval.path=$CKPT" \
    "common_eval.results_path=$WORK/results" common_eval.quiet=false \
    dataset.gen_subset=train "+task.target_dictionary=$DICT_DIR"

# ---- Step 3/3: 组装输出 + 可选热词修正 ----
echo "[transcribe] === Step 3/3: 输出 ==="
"$PYTHON_BIN" "$SUBMISSION/src/make_predictions.py" \
    --infer_log "$WORK/results/train/infer.log" \
    --input_manifest "$MANIFEST" --output_path "$WORK/asr_raw.jsonl"

FINAL="$WORK/asr_raw.jsonl"
if [[ $NO_HOTWORD -eq 0 ]]; then
    if [[ ! -f "$HOTWORD_FIX" || ! -f "$HOTWORD_DICT" ]]; then
        echo "[ERROR] 热词件缺失: $HOTWORD_FIX / $HOTWORD_DICT"
        echo "        （补齐后重跑，或用 --no-hotword 跳过）"
        exit 1
    fi
    HOTWORD_DICT="$HOTWORD_DICT" "$PYTHON_BIN" -X utf8 "$HOTWORD_FIX" "$WORK/asr_raw.jsonl" "$WORK/final.jsonl"
    FINAL="$WORK/final.jsonl"
fi

if [[ -n "$OUTPUT" ]]; then
    cp "$FINAL" "$OUTPUT"
    echo "[transcribe] 已写入: $OUTPUT"
fi

# 打印 id<TAB>text 到 stdout
"$PYTHON_BIN" -X utf8 - "$FINAL" <<'PYEOF'
import json, sys
for l in open(sys.argv[1], encoding='utf-8'):
    o = json.loads(l)
    print(f"{o['id']}\t{o['text']}")
PYEOF
