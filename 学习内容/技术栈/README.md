# TeleSpeech-ASR 方言导航语音识别 — 完整技术栈

---

## 一、深度学习引擎

### PyTorch 1.13.0

| 组件 | 版本 | 用途 |
|------|------|------|
| torch | 1.13.0+cu117 | 张量计算、自动微分、GPU 加速 |
| torchaudio | 0.13.0+cu117 | MFCC 特征提取、音频 I/O、重采样 |
| torchvision | 0.14.0+cu117 | 图像模型相关（预训练模型依赖） |
| CUDA | 13.0 | GPU 计算后端 |
| cuDNN | 随 CUDA | 深度卷积加速 |

### GPU 硬件

| 参数 | 值 |
|------|-----|
| 型号 | NVIDIA A100-SXM4-80GB |
| 显存 | 81,920 MiB |
| 架构 | Ampere (SM 8.0) |
| 驱动 | 580.95.05 |
| 功耗 | 400W TDP |
| 精度 | FP32 / FP16 / BF16 / TF32 |

---

## 二、训练框架 Fairseq

### 核心模块

| 模块 | 路径 | 功能 |
|------|------|------|
| `hydra_train.py` | `fairseq_cli/hydra_train.py` | Hydra 配置驱动的训练入口 |
| `Wav2VecCtc` | `fairseq/models/wav2vec/wav2vec2_asr.py` | CTC 语音识别模型 |
| `Wav2VecEncoder` | 同上 | 编码器（特征提取 + Transformer） |
| `fairseq_task.py` | `fairseq/tasks/fairseq_task.py` | 任务基类 |
| `checkpoint_utils.py` | `fairseq/checkpoint_utils.py` | 模型保存/加载 |
| `dictionary.py` | `fairseq/data/dictionary.py` | 字典映射（文字↔ID） |
| `ctc.py` | `fairseq/criterions/ctc.py` | CTC Loss 计算 |

### 自研模块 (data2vec_dialect)

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `infer.py` | 518 | 推理主脚本，加载模型→推理→计算 WER |
| `config/infer.yaml` | — | 推理配置 |
| `config/v2_dialect_asr/base_audio_finetune_140h.yaml` | — | 训练配置 (max_tokens/lr/optimizer) |
| `data/spec_dataset.py` | 500+ | 数据加载，读取 Kaldi ARK 格式 |
| `tasks/audio_finetuning.py` | 400+ | CTC 微调任务定义 |
| `models/wav2vec2.py` | 600+ | Data2Vec 多模态模型 |
| `run_scripts/run_d2v_finetune.sh` | 20行 | 训练启动脚本 |
| `run_scripts/decode.sh` | 15行 | 推理启动脚本 |

### Hydra 配置体系

```yaml
# base_audio_finetune_140h.yaml
common:
  fp16: false          # 全精度训练
  log_interval: 200    # 每200步输出loss

dataset:
  max_tokens: 50000    # 每batch最大token数
  train_subset: train  # 训练集标识
  valid_subset: dev    # 验证集标识

model:
  _name: wav2vec_ctc   # 模型类型
  w2v_path: ???        # 预训练编码器路径
  activation_dropout: 0.1
  freeze_finetune_updates: 0  # 冻结步数

optimization:
  max_update: 40000    # 总训练步数
  lr: [1e-06]          # 学习率
  update_freq: [1]     # 梯度累积

criterion:
  _name: ctc           # CTC损失函数

optimizer:
  _name: adam
  adam_betas: (0.9, 0.98)
  adam_eps: 1e-08

lr_scheduler:
  _name: tri_stage     # 三阶段学习率
  phase_ratio: [0.1, 0.4, 0.5]
```

---

## 三、模型架构 Data2Vec 2.0

### 编码器 (Wav2VecEncoder)

```
输入: MFCC特征 (T帧, 40维)
│
├── ConvFeatureExtractionModel (特征前端)
│   ├── Conv1d(40→512, kernel=3, stride=2) + Dropout + LayerNorm + GELU
│   └── Conv1d(512→512, kernel=3, stride=2) + Dropout + LayerNorm + GELU
│   输出: (T/4帧, 512维)
│
├── LayerNorm + Linear(512→768)
│
├── RelativePositionalEncoder (5层 Conv1d, width=95, groups=16)
│
├── Transformer Encoder (12层, 768维)
│   ├── Multi-Head Self-Attention (12头)
│   │   └── AltAttention (优化的注意力实现)
│   ├── LayerNorm
│   └── MLP (768→3072→768, GELU)
│
└── 输出: (T/4帧, 768维)
```

### 完整模型 (Wav2VecCtc)

```
Wav2VecEncoder 输出 (帧, 768)
    ↓
Linear(768 → 7531)    ← CTC分类头
    ↓
(帧, 7531) 每帧每个字符的概率
    ↓
CTC Viterbi 解码
    ↓
文字输出
```

### 字典 (dict.chr7531.txt)

```
0 → <blank>  (CTC blank token, id=0)
1 → <unk>    (未知字符, id=1)
是 → 2
好 → 3
...
共 7531 个字符
```

---

## 四、数据处理管线

### MFCC 特征提取

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 16,000 Hz | |
| 帧长 | 25ms (400采样点) | |
| 帧移 | 10ms (160采样点) | |
| FFT 点数 | 512 | |
| Mel 滤波器数 | 40 | 输出维度 |
| 低频截止 | 40 Hz | |
| 高频截止 | 7,800 Hz | Nyquist-200 |
| 窗函数 | Hamming | 减少频谱泄漏 |
| 工具 | torchaudio.transforms.MFCC | |

### 数据增强

| 方法 | 工具 | 参数 | 效果 |
|------|------|------|------|
| 速度扰动 | SoX tempo | 0.9x / 1.1x | 数据量×3 |
| (MFCC已离线提取，训练时无在线增强) | | | |

### data.list 格式

```
utt:G0039_0_S1001<TAB>feat:/root/.../mfcc.1.ark:17<TAB>feat_shape:690,40<TAB>text:我明天上午9点...<TAB>token:我 明 天 上 午...<TAB>tokenid:[5858,5669,...]<TAB>token_shape:25,7531
```

---

## 五、环境与基础设施

### 容器环境

| 项目 | 说明 |
|------|------|
| 容器运行时 | Containerd + Kubernetes |
| 操作系统 | Ubuntu 22.04 (容器镜像) |
| 文件系统 | OverlayFS (50G) + NVMe XFS (3.5T) + 网络 (700T) |
| 数据盘挂载 | `/localdisk-tmp` (NVMe 3.5T), `/Save` (网络 700T) |

### Conda 环境 (chuanyu-ASR)

| 包 | 版本 | 用途 |
|------|------|------|
| python | 3.8.20 | |
| pip | — | 包管理 |
| torch | 1.13.0+cu117 | 深度学习 |
| torchaudio | 0.13.0+cu117 | 音频处理 |
| torchvision | 0.14.0+cu117 | |
| fairseq | 0.12.2 (源码安装) | 训练框架 |
| kaldiio | 2.18.1 | Kaldi ARK 读写 |
| timm | 1.0.28 | 视觉模型库 |
| soundfile | 0.13.1 | WAV 读写 |
| editdistance | 0.8.1 | 编辑距离 |
| sentencepiece | — | BPE分词 |
| tensorboardX | — | 训练可视化 |

### 环境变量

```bash
PYTHONPATH="/root/asr-competition/fairseq:$PWD:$PYTHONPATH"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

## 六、训练配置演化

| 版本 | 起点 | 数据 | LR | 冻结 | 结果 |
|------|------|:--:|------|:--:|:--:|
| v1 | large.pt | 35k | 1e-05 | 0 | 15.50% |
| v4 | kespeech_encoder | 35k | 5e-06 | 10k | 7.08% |
| v8 | finetune_large_kespeech | 35k | 1e-06 | 0 | 5.50% |
| nav80k | finetune_large_kespeech | 13k增强 | 1e-06 | 0 | **0.70%** |

## 七、评测体系

### 官方工具（比赛）

| 工具 | 功能 |
|------|------|
| `normalize.py` | 文本规范化：去标点、只保留中文/字母/数字 |
| `score.py` | CER 计算：`(S + D + I) / N`，字符级编辑距离 |

### 提交流程

```
测试WAV → MFCC提取 → infer.py(nav80k) → HYPO输出 
→ normalize.py(JSONL) → predictions.jsonl → 提交
```
