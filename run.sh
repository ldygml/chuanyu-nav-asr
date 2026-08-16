#!/bin/bash
# ============================================================
# CCF IVC 2026 — 智能导航方言语音识别挑战赛 决赛提交
# run.sh — 离线推理入口，接受组委会统一调用
#
# 调用方式:
#   bash run.sh --input_manifest /data/input/manifest.jsonl \
#               --output_path /data/output/predictions.jsonl \
#               --device cuda:0
# ============================================================
set -euo pipefail

# ---- 1. 解析参数 ----
INPUT_MANIFEST=""
OUTPUT_PATH=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input_manifest) INPUT_MANIFEST="$2"; shift 2 ;;
        --output_path)    OUTPUT_PATH="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT_MANIFEST" || -z "$OUTPUT_PATH" ]]; then
    echo "[ERROR] 用法: run.sh --input_manifest <FILE> --output_path <FILE> [--device cuda:N]"
    exit 1
fi

echo "[run.sh] input_manifest=$INPUT_MANIFEST"
echo "[run.sh] output_path=$OUTPUT_PATH"
echo "[run.sh] device=$DEVICE"

# ---- 2. 路径设置 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/tmp/submission_$$"
mkdir -p "$WORK_DIR"
echo "[run.sh] work_dir=$WORK_DIR"

# ---- 3. 定位 Python ----
# 优先级: 环境变量 > conda chuanyu-ASR > 系统 conda > 系统 python3
if [[ -z "${PYTHON_BIN:-}" ]]; then
    # 优先使用 conda 环境（自带 pip + PyTorch）
    if [[ -x /opt/conda/envs/chuanyu-ASR/bin/python3 ]]; then
        PYTHON_BIN="/opt/conda/envs/chuanyu-ASR/bin/python3"
    elif [[ -x /opt/conda/bin/python3 ]]; then
        PYTHON_BIN="/opt/conda/bin/python3"
    elif command -v python3 &>/dev/null; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python &>/dev/null; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "[ERROR] 找不到 Python，请设置 PYTHON_BIN 环境变量"
        exit 1
    fi
fi
echo "[run.sh] python=$PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# ---- 4. 离线安装依赖 ----
# 组委会评测环境禁止联网，从本地 wheels/ 离线安装所有依赖
# 若检测到 chuanyu-ASR conda 环境则跳过（已完整预装）
WHEELS_DIR="$SCRIPT_DIR/wheels"
if [[ "$PYTHON_BIN" == *chuanyu-ASR* ]]; then
    echo "[run.sh] 检测到 chuanyu-ASR conda 环境，跳过依赖安装"
elif [[ -d "$WHEELS_DIR" ]] && ls "$WHEELS_DIR"/*.whl &>/dev/null; then
    echo "[run.sh] === 离线安装依赖 ==="

    # 检查 PyTorch 是否已预装
    if $PYTHON_BIN -c "import torch" &>/dev/null; then
        echo "[run.sh] PyTorch 已预装，仅安装其他依赖"
        TMP_WHEELS="$WORK_DIR/wheels_subset"
        mkdir -p "$TMP_WHEELS"
        for whl in "$WHEELS_DIR"/*.whl; do
            case "$(basename "$whl")" in
                torch-*|torchaudio-*|torchvision-*|numpy-*) ;;
                *) cp "$whl" "$TMP_WHEELS/" ;;
            esac
        done
        $PYTHON_BIN -m pip install --no-index --no-deps "$TMP_WHEELS"/*.whl
        rm -rf "$TMP_WHEELS"
    else
        echo "[run.sh] PyTorch 未预装，从 wheels 完整安装"
        $PYTHON_BIN -m pip install --no-index --no-deps "$WHEELS_DIR"/*.whl
    fi

    echo "[run.sh] 验证依赖..."
    $PYTHON_BIN -c "
import torch, torchaudio, kaldiio, soundfile, sentencepiece
print(f'  torch={torch.__version__}, torchaudio={torchaudio.__version__}')
" || { echo "[ERROR] 核心依赖安装失败"; exit 1; }
else
    echo "[run.sh] 未找到 wheels/，假设依赖已预装"
fi

# ---- 5. 环境变量 ----
export PYTHONPATH="$SCRIPT_DIR/fairseq:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}"

# ---- 6. 提取 MFCC 并生成 data.list ----
echo "[run.sh] === Step 1/3: 提取 MFCC 特征 ==="
$PYTHON_BIN "$SCRIPT_DIR/src/prepare_manifest_data.py" \
    --input_manifest "$INPUT_MANIFEST" \
    --output_dir "$WORK_DIR/data"

# ---- 7. 模型推理 ----
echo "[run.sh] === Step 2/3: 模型推理 ==="
$PYTHON_BIN "$SCRIPT_DIR/src/infer.py" \
    --config-dir "$SCRIPT_DIR/src/config" \
    --config-name infer \
    task=spec_finetuning \
    "task.data=$WORK_DIR/data" \
    task.normalize=false \
    "common.user_dir=$SCRIPT_DIR/src" \
    "common_eval.path=$SCRIPT_DIR/weights/model.pt" \
    "common_eval.results_path=$WORK_DIR/results" \
    common_eval.quiet=false \
    dataset.gen_subset=train \
    "+task.target_dictionary=$SCRIPT_DIR/weights"

# ---- 8. 生成 predictions.jsonl ----
echo "[run.sh] === Step 3/3: 生成预测文件 ==="
$PYTHON_BIN "$SCRIPT_DIR/src/make_predictions.py" \
    --infer_log "$WORK_DIR/results/train/infer.log" \
    --input_manifest "$INPUT_MANIFEST" \
    --output_path "$OUTPUT_PATH"

# ---- 9. 清理临时文件 ----
rm -rf "$WORK_DIR"

echo "[run.sh] 完成: $OUTPUT_PATH"
exit 0
