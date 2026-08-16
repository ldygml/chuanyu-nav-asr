# 川渝方言导航语音识别（CCF IVC 2026 赛项三）

队伍 **LynRose_Enigma** 的参赛完整工程：基于 **Data2Vec 2.0 + CTC** 的川渝方言导航语音识别系统。

- **比赛**：CCF IVC 2026 智能导航方言语音识别挑战赛（川渝方言导航语音）
- **成绩**：初赛第 14 名（A 榜纯模型基线 CER = 6.68%，热词后处理版提交）
- **模型**：Data2Vec 2.0 编码器 + CTC，导航数据增强 + 多段续训（nav80k → nav80k_cont）
- **仓库**：完整推理代码 + fairseq + 热词后处理 + 训练脚本 + 环境清单（权重在 Hugging Face）

---

## 目录结构

```
├── transcribe.sh          # ★通用转写工具：wav/目录 → 文字，一条命令
├── hotword_fix.py         # 热词后处理（纯字典替换，1981 对词典）
├── 词典/                  # 热词词典（hotword_dict_final.md，合并去重版）
├── src/                   # ASR 代码（data2vec_dialect：模型/任务/推理三件套）
├── fairseq/               # fairseq 源码（本项目依赖的版本）
├── scripts/               # 训练/推理脚本（含 nav80k_cont 续训命令）
├── docs/                  # 方案文档、环境配置文档、数据集制作流程等
├── dict.chr7531.txt       # 字符词典（7531 字）
├── requirements.txt       # 环境依赖清单（与训练服务器一致）
└── PIPELINE说明.md        # 完整 pipeline 说明（训练路线/推理流程/热词后处理）
```

---

## 环境安装

- 系统：Linux（x86_64），CUDA 11.7，Python 3.8
- 参考命令：

```bash
# 1) conda 环境
conda create -n chuanyu-ASR python=3.8 -y && conda activate chuanyu-ASR

# 2) PyTorch（CUDA 11.7 预编译）
pip install torch==1.13.0+cu117 torchaudio==0.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 3) 其余依赖
pip install -r requirements.txt
```

> 完全离线安装可选手动打包的 wheels（见比赛提交包，未入仓库）。

## 模型权重

权重文件较大（约 3.8 GB），托管在 Hugging Face：[999ffgml/chuanyu-nav-asr](https://huggingface.co/999ffgml/chuanyu-nav-asr)

```bash
# 下载模型 + 词典到 weights/
pip install huggingface_hub
hf download 999ffgml/chuanyu-nav-asr --local-dir weights
```

> 国内网络可先执行 `export HF_ENDPOINT=https://hf-mirror.com`（镜像加速）再下载。

```
weights/
├── model.pt             # nav80k_cont checkpoint（最终提交模型）
└── dict.chr7531.txt     # 字符词典
```

---

## 快速开始：语音转文字

```bash
# 单条音频
bash transcribe.sh --input /path/to/audio.wav

# 整个目录（*.wav）
bash transcribe.sh --input /path/to/wav_dir --output result.jsonl

# 跳过热词修正 / 指定 GPU
bash transcribe.sh --input audio.wav --no-hotword --device cuda:1
```

- 输入：任意采样率、任意声道数的 WAV（自动重采样 16k 单声道）
- 输出：stdout 打印 `id<TAB>文字`；`--output` 同时写 jsonl（`{"id","text"}`）
- 热词修正默认开启（`词典/hotword_dict_final.md`，1981 对）；`--no-hotword` 跳过
- 环境变量：`NAV_ASR_CKPT`（权重路径，默认 `weights/model.pt`）、`NAV_ASR_PYTHON`（解释器）

### 底层推理三步（等价操作）

```
wav → prepare_manifest_data.py（40 维 MFCC + data.list）
    → infer.py（hydra，viterbi greedy 解码）
    → make_predictions.py（按 IDX 排序、去 <unk> → jsonl）
```

---

## 训练复现（路线概述）

```
基线               30.70%    finetune_large_kespeech 无微调
全模型微调+超低LR   5.50%     1e-06，21k 步
纯导航数据          6.28%     导航数据 3,385 条，单段 40k
SoX 速度扰动增强     4.93%     0.9x/1.1x ×4，13,240 条
二段续训           1.88%     40k → 60k
三段续训           1.10%     60k → 80k（独立 300 条测试）
A 榜首测            6.69%     dev 与 A 榜分布差异 ~5.5pp
nav80k 续训        6.68%     原参数续训 20k（A 榜 CER，两次推理逐字复现）
```

- 核心三项技术：全模型微调 + 超低 LR 1e-06、SoX 速度扰动、多段续训（每段从最优 checkpoint 出发 + 重置 optimizer）
- 训练命令见 `scripts/`（含 nav80k_cont 续训脚本）；数据增强见 `docs/自用数据集制作流程.md`
- 失败的探索（勿复现）：混合通用数据续训（navmix）、加噪 + SpecAugment 翻倍（nav_specaug）、FastCorrect 纠错模型微调（FC，过拟合）

## 热词后处理

- 方案：**纯字典替换**（按错词长度降序 find 替换 + 子串防护 + REVERT_SET + 上下文例外），比音素模糊匹配（asr-hotword）可控得多
- 词典来源：全部为测试集之外的外部地名/机构名资料与常识整理
- 原理与迭代历程详见 `PIPELINE说明.md` 第 4 节

## 致谢

- 官方基线：TeleSpeech-ASR（Data2Vec 2.0 方言 ASR）
- 预训练底座：KeSpeech 系列（large / finetune_large_kespeech）
- 热词库早期原型：HaujetZhao/asr-hotword
