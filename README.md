# chuanyu-nav-asr

川渝方言导航语音识别系统（CCF IVC 2026 智能导航方言语音识别挑战赛参赛项目，初赛第 14 名）。

基于 **Data2Vec 2.0 + CTC**，输入任意导航语音 WAV，输出汉字文本。A 榜纯模型基线 CER = **6.68%**。

- **模型**：nav80k_cont（约 3.8 GB），托管于 Hugging Face
- **环境**：Linux x86_64 · Python 3.8 · CUDA 11.7
- **开箱即用**：一条命令完成语音转文字，自动重采样、自动热词修正

> **TeleSpeech Inside**
>
> 本项目是基于 [TeleSpeech 模型](https://github.com/Tele-AI/TeleSpeech-ASR)（Data2Vec 2.0 方言 ASR）二次开发的模型衍生品：
> [src/](src/) 派生自其 `data2vec_dialect`，模型权重 nav80k_cont 为对
> TeleSpeech-ASR1.0-large-kespeech 微调续训的产物。
>
> **主要修改**：MFCC 提取改为纯 Python 三件套（prepare_manifest_data.py）、增加输出组装
> （make_predictions.py）、捆绑 fairseq 0.12.2 使仓库自足、增加热词后处理与一键转写入口
> （transcribe.sh）。
>
> 使用本项目须遵守《TeleSpeech 模型社区许可协议》：
> **仅限非商业用途**；商用须事先向许可方（tele_ai@chinatelecom.cn）登记并获书面授权。
> 协议副本见 [third_party_licenses/LICENSE_TeleSpeech.pdf](third_party_licenses/LICENSE_TeleSpeech.pdf)。

---

## 快速开始

### 1. 安装环境

支持两种环境（依赖版本由 pip 按 Python 版本自动解析，均实测可跑通推理）：

**路线 A：Python 3.8 + CUDA 11.7**（旧显卡/旧驱动）

```bash
conda create -n chuanyu-ASR python=3.8 -y && conda activate chuanyu-ASR

pip install torch==1.13.0+cu117 torchaudio==0.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
```

**路线 B：Python 3.10+ + torch 2.x**（新机器/新驱动，或使用机器预装 torch）

```bash
conda create -n chuanyu-ASR python=3.10 -y && conda activate chuanyu-ASR

pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> torchvision 必须与 torch 一起装：requirements.txt 中 timm 依赖未钉版本的
> torchvision，单独 `pip install -r` 会拉最新 torchvision 并连锁升级 torch，
> 导致与已装 torchaudio 版本错位。

两条路线最后都要编译 fairseq 的 Cython 扩展：

```bash
cd fairseq && python setup.py build_ext --inplace && cd ..
```

> 国内网络 pip 可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
> 路线 B 实测环境：Python 3.12 + torch 2.12（RTX 4090）

### 2. 下载模型权重

```bash
pip install huggingface_hub
mkdir weights
hf download 999ffgml/chuanyu-nav-asr --local-dir weights
```

> 国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`（镜像加速）

完成后应得到：

```
weights/
├── model.pt             # nav80k_cont checkpoint（约 3.8 GB）
└── dict.chr7531.txt     # 字符词典（7531 字）
```

### 3. 转写语音

```bash
bash transcribe.sh --input 你的音频.wav
```

stdout 打印 `id<TAB>文字`，例如：

```
audio_001	前方五百米左转进入天府大道
```

`--input` 传目录即可批量转写目录内所有 `*.wav`。

---

## transcribe.sh 用法

| 参数 | 说明 |
|---|---|
| `--input <wav 或 目录>` | 必填。任意采样率/声道数的 WAV，自动重采样 16k 单声道 |
| `--output <file.jsonl>` | 可选。同时写 `{"id","text"}` 格式结果文件 |
| `--no-hotword` | 跳过热词修正（默认开启） |
| `--device cuda:N` | 推理 GPU（默认 cuda:0） |

环境变量覆盖：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `NAV_ASR_CKPT` | 模型 checkpoint 路径 | `weights/model.pt` |
| `NAV_ASR_PYTHON` | Python 解释器 | 自动探测（conda chuanyu-ASR 或系统 python3） |
| `HOTWORD_DICT` | 热词词典路径 | `hotword/hotword_dict_final.md` |

### 底层推理流程

```
wav → prepare_manifest_data.py（重采样 → 40 维 MFCC）
    → infer.py（fairseq/hydra，viterbi greedy 解码）
    → make_predictions.py（按 IDX 排序、去 <unk>）
    → hotword_fix.py（可选，字典热词修正）
```

---

## 热词后处理

纯字典替换方案，词典 `hotword/hotword_dict_final.md`（**1981 对**），格式为每行 `错误词 正确词`（空格分隔，`#` 开头为注释）。

- 按错词长度降序替换 + 子串防护 + 手动例外（REVERT_SET / PREV_EXCEPT），误伤率低
- **自定义词典**：直接编辑词典文件，或 `export HOTWORD_DICT=/path/to/your_dict.md` 后运行 transcribe.sh
- 算法细节与质量验证见 [PIPELINE.md](PIPELINE.md) 第 4 节

---

## 训练复现

训练路线（dev / A 榜 CER）：

```
基线               30.70%    finetune_large_kespeech 无微调
全模型微调+超低LR   5.50%     1e-06，21k 步
纯导航数据          6.28%     导航数据 3,385 条，单段 40k
SoX 速度扰动增强     4.93%     0.9x/1.1x ×4，13,240 条
二段续训           1.88%     40k → 60k
三段续训           1.10%     60k → 80k
A 榜首测            6.69%     dev 与 A 榜分布差异 ~5.5pp
nav80k 续训        6.68%     原参数续训 20k（A 榜 CER，两次推理逐字复现）
```

- 训练脚本见 `scripts/`（含 `train_nav80k_cont.sh`）；详细说明见 [PIPELINE.md](PIPELINE.md) 第 2 节
- 失败的探索（勿复现）：navmix（混合通用数据续训）、nav_specaug（加噪 + SpecAugment）、FastCorrect 纠错模型微调

> 训练数据（比赛官方数据集 + 自制增强集）受比赛数据协议约束，未随仓库发布。

---

## 目录结构

```
├── transcribe.sh          # 通用转写工具（唯一使用入口）
├── hotword_fix.py         # 热词后处理（纯字典替换）
├── hotword/               # 热词词典（hotword_dict_final.md，1981 对）
├── src/                   # ASR 代码（data2vec_dialect 三件套 + 任务定义）
├── fairseq/               # 依赖的 fairseq 0.12.2 源码
├── scripts/               # 训练/数据处理脚本
├── dict.chr7531.txt       # 字符词典（7531 字）
├── requirements.txt       # Python 依赖清单
├── PIPELINE.md            # 训练路线 / 推理流程 / 热词算法详解
└── submission.yaml        # 竞赛提交元数据（权重 SHA-256 等）
```

---

## 致谢

- 官方基线：TeleSpeech-ASR（Data2Vec 2.0 方言 ASR）
- 预训练底座：KeSpeech 系列（large / finetune_large_kespeech）
