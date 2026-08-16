# TeleSpeech-ASR 推理流程详解

> 基于 Tele-AI/TeleSpeech-ASR 项目，使用 `finetune_large_kespeech.pt` 模型进行方言语音识别推理

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        推理管线 (infer.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌───────┐ │
│  │ data.list │───▶│ SpecDataset   │───▶│ Wav2VecCtc   │───▶│ 文本  │ │
│  │ (MFCC特征) │    │ (数据加载+批处理)│    │ (神经网络)    │    │ 输出  │ │
│  └──────────┘    └───────────────┘    └──────────────┘    └───────┘ │
│                         │                     │                     │
│                    kaldiio.load_mat      Transformer Encoder         │
│                    读取 .ark 文件         + CTC 线性投影层             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、数据准备（推理前）

### 2.1 必需文件

推理需要以下文件，全部位于 `/root/asr-competition/test_data/` 目录下：

| 文件 | 格式 | 说明 |
|------|------|------|
| `data.list` | TSV | 核心索引文件，每行包含一条语音的全部元信息 |
| `feat/mfcc.scp` | Kaldi SCP | utterance ID → ark 文件路径+偏移量映射 |
| `feat/mfcc.*.ark` | Kaldi ARK | 预提取的 40 维 MFCC 特征（二进制） |
| `feat/feat2len.txt` | TXT | 每条 utterance 的帧数 |
| `text` | TXT | utterance ID → 参考转录文本（用于计算 WER） |
| `train.tsv` | 软链接 → data.list | Fairseq 要求的数据子集文件 |
| `dev.tsv` | 软链接 → data.list | 验证子集（推理时通常也用同一份） |

### 2.2 data.list 格式

`data.list` 是推理的核心输入文件，以 `\t`（Tab）分隔字段。每行一条语音：

```
utt:<utterance_id>	feat:<ark_path:byte_offset>	feat_shape:<frames>,40	text:<参考文本>	token:<空格分隔的字>	tokenid:[id1,id2,...]	token_shape:<字数>,7531
```

**实际示例**（一行）：

```
utt:1000005_0e4c205e	feat:/root/asr-competition/test_data/feat/1/mfcc.1.ark:17	feat_shape:690,40	text:该行将推出无卡化时代电子支付	token:该 行 将 推 出 无 卡 化 时 代 电 子 支 付	tokenid:[5858,5669,1620,2388,502,2586,651,616,2601,176,3975,1523,2517,168]	token_shape:14,7531
```

**字段解释**：

| 字段 | 示例值 | 说明 |
|------|--------|------|
| `utt` | `1000005_0e4c205e` | 唯一 utterance ID |
| `feat` | `feat/1/mfcc.1.ark:17` | MFCC 特征文件路径 + 字节偏移量 |
| `feat_shape` | `690,40` | (帧数, 特征维度) |
| `text` | `该行将推出无卡化时代电子支付` | 参考转录（推理时用于对比计算错误率） |
| `token` | `该 行 将 推 出 ...` | 空格分隔的中文字符 |
| `tokenid` | `[5858,5669,1620,...]` | 每个字在字典中的 ID（字典共 7531 个字符） |
| `token_shape` | `14,7531` | (字数, 字典大小) |

### 2.3 MFCC 特征

- **维度**：40 维 MFCC（Mel-Frequency Cepstral Coefficients）
- **采样率**：16kHz
- **帧移**：10ms，**帧长**：25ms
- **提取工具**：使用 `torchaudio.transforms.MFCC` 或 Kaldi `compute-mfcc-feats`
- **存储格式**：Kaldi ARK 二进制格式，通过 `kaldiio.load_mat()` 读取

### 2.4 字典

模型使用的字典 `dict.chr7531.txt` 包含 7531 个中文字符：

```
0 2      # 字符0 → ID 2
1 3      # 字符1 → ID 3
是 2     # "是" → ID 2
好 3     # "好" → ID 3
...
```

额外 token：
- `<blank>` (CTC blank token)：ID 0
- `<unk>`：ID 1
- `<sos/eos>`：ID 5536

---

## 三、模型架构

### 3.1 模型结构图

```
Wav2VecCtc (finetune_large_kespeech.pt)
│
├── Wav2VecEncoder
│   └── Data2VecMultiModel (0.3B 参数)
│       ├── AudioEncoder
│       │   ├── ConvFeatureExtractionModel (特征前端)
│       │   │   ├── Conv1d(40 → 512, kernel=3, stride=2)  ← 降采样 2x
│       │   │   ├── LayerNorm + GELU
│       │   │   ├── Conv1d(512 → 512, kernel=3, stride=2) ← 降采样 2x
│       │   │   └── LayerNorm + GELU
│       │   │   (总降采样：4x，如 10s 音频 → ~250 帧)
│       │   │
│       │   ├── project_features: Linear(512 → 768)
│       │   ├── RelativePositionalEncoder (5层 Conv1d, 95宽度)
│       │   └── Transformer Encoder (12 层)
│       │       ├── Multi-Head Self-Attention (12 头)
│       │       ├── LayerNorm + MLP (768 → 3072 → 768)
│       │       ├── GELU 激活
│       │       └── Dropout + LayerDrop
│       │
│       └── (其他模态: IMAGE, TEXT — 预训练时使用，推理时不用)
│
└── CTC Projection Layer
    └── Linear(768 → 7531)  ← 输出每个字+blank的概率
```

### 3.2 关键参数

| 参数 | 值 |
|------|-----|
| 总参数量 | 0.3B (3亿) |
| 编码器层数 | 12 |
| 注意力头数 | 12 |
| 隐藏维度 | 768 |
| FFN 维度 | 3072 |
| 输出维度 | 7531 (字典大小) |
| 预训练数据 | 30万小时无标注多方言语音 |
| 微调数据 | KeSpeech 8 种方言 |

---

## 四、推理执行流程（代码级别）

### 4.1 入口函数

```python
# infer.py:446-480
def hydra_main(cfg: InferConfig) -> float:
    """Hydra 配置驱动的推理入口"""
    # 1. 解析配置 (YAML + 命令行覆盖)
    # 2. 调用 main(cfg)
    # 3. 返回 WER 值
```

### 4.2 main() 函数流程

```python
# infer.py:391-445
def main(cfg: InferConfig) -> float:
    # Step 1: 创建推理处理器
    processor = InferenceProcessor(cfg)
    
    # Step 2: 获取数据迭代器
    data_itr = processor.get_dataset_itr()
    
    # Step 3: 逐 batch 处理
    for sample in data_itr:
        processor.process_sample(sample)
    
    # Step 4: 计算并输出 WER
    wer = processor.total_errors / processor.total_length * 100
    logger.info(f"Word error rate: {wer:.4f}")
    return wer
```

### 4.3 InferenceProcessor 初始化

```python
# infer.py:101-160
class InferenceProcessor:
    def __init__(self, cfg):
        # 1. 设置任务 (spec_finetuning)
        self.task = tasks.setup_task(cfg.task)
        
        # 2. 加载模型
        #    - 如果模型中有 adapter，加载 adapter 权重
        #    - 否则直接加载整个模型
        models, saved_cfg = self.load_model_ensemble()
        
        # 3. 移动到 GPU
        self.models = [model.cuda() for model in models]
        
        # 4. 设为 eval 模式
        for model in self.models:
            model.eval()
        
        # 5. 初始化计数器
        self.total_errors = 0   # 编辑距离错误总数
        self.total_length = 0   # 参考文本总长度
        self.generator = self.task.build_generator(models, cfg)
```

### 4.4 数据加载 — SpecDataset

```python
# spec_dataset.py:398-440
class SpecDataset:
    def __getitem__(self, index):
        # 1. 从 data.list 获取文件路径
        fn = self.fnames[index]
        
        # 2. 用 kaldiio 读取 MFCC ark 文件
        feats = kaldiio.load_mat(path_or_fp)
        # feats shape: (frames, 40)
        
        # 3. 转成 PyTorch tensor
        feats = torch.from_numpy(feats).float()  # (T, 40)
        
        # 4. 后处理 (normalize / crop)
        feats = self.postprocess(feats)
        
        return {"id": index, "source": feats}
```

### 4.5 批处理 — collater

```python
# spec_dataset.py:337-395
def collater(self, samples):
    # 1. 收集所有样本，padding 到相同长度
    sources = [s["source"] for s in samples]
    max_len = max(s.shape[0] for s in sources)
    
    # 2. 创建 (batch, max_frames, 40) 的 padded tensor
    feats = torch.zeros(len(samples), max_len, 40)
    for i, src in enumerate(sources):
        feats[i, :src.shape[0]] = src
    
    # 3. 创建 padding mask
    padding_mask = torch.ones(len(samples), max_len, dtype=torch.bool)
    for i, src in enumerate(sources):
        padding_mask[i, :src.shape[0]] = False
    
    return {
        "id": ids,
        "net_input": {"source": feats, "padding_mask": padding_mask},
        "target": target_tokens,  # 参考 token IDs
    }
```

### 4.6 模型前向传播

```python
# wav2vec2_asr.py:588-622
class Wav2VecEncoder:
    def forward(self, source, padding_mask, **kwargs):
        # source: (batch, frames, 40)
        
        # 1. 卷积特征提取 (降采样 4x)
        x = self.feature_extractor(source)
        # x: (batch, frames/4, 512)
        
        # 2. 线性投影到模型维度
        x = self.project_features(x)
        # x: (batch, frames/4, 768)
        
        # 3. 添加相对位置编码
        x = self.relative_positional_encoder(x)
        
        # 4. Transformer Encoder (12 层自注意力)
        x = self.encoder(x, padding_mask)
        # x: (batch, frames/4, 768)
        
        return {"encoder_out": x, "padding_mask": padding_mask}

# wav2vec2_asr.py:210-220
class Wav2VecCtc:
    def forward(self, **kwargs):
        # 1. 编码
        encoder_out = self.w2v_encoder(**kwargs)
        
        # 2. CTC 线性投影
        logits = self.proj(encoder_out["encoder_out"])
        # logits: (batch, frames/4, 7531)
        
        return {"out": logits, "padding_mask": ...}
```

### 4.7 CTC 解码

```python
# 在 task.inference_step 中
def decode_ctc(logits, dictionary):
    # logits: (batch, frames, 7531)
    
    # 1. Viterbi 解码：每帧取最大概率的 token
    pred_ids = logits.argmax(dim=-1)  # (batch, frames)
    
    # 2. CTC 去重：移除连续重复 token
    unique_ids = []
    for i in range(len(pred_ids)):
        if i == 0 or pred_ids[i] != pred_ids[i-1]:
            unique_ids.append(pred_ids[i])
    
    # 3. 移除 blank token (ID=0)
    decoded = [id for id in unique_ids if id != 0]
    
    # 4. ID → 文字
    text = dictionary.string(decoded)
    # 例如 [5858, 5669, 1620, ...] → "该行将推..."
    
    return text
```

### 4.8 计算错误率

```python
# infer.py:290-340
def process_sentence(self, sample, hypo, sid, batch_id):
    # 1. 获取预测文字 (hypothesis)
    hyp_tokens = hypo["tokens"]       # 预测的 token IDs
    hyp_text = dictionary.string(hyp_tokens)  # → "该行将推..."
    
    # 2. 获取参考文字 (reference)
    ref_tokens = sample["target"]     # 参考 token IDs
    ref_text = dictionary.string(ref_tokens)  # → "该行将推出..."
    
    # 3. 计算编辑距离
    errors = editdistance.eval(hyp_text, ref_text)
    length = len(ref_text)
    
    # 4. 累积 (所有 batch 累加后计算 WER)
    return errors, length

# 最终 WER = total_errors / total_length * 100
```

---

## 五、解码配置

### 5.1 infer.yaml 配置

```yaml
# @package _group_
defaults:
    - task: null
    - model: null

hydra:
  run:
    dir: ${common_eval.results_path}/${dataset.gen_subset}

common_eval:
  results_path: null        # 结果输出路径
  path: null                # 模型路径
  post_process: letter      # 后处理方式：字符级

dataset:
  max_tokens: 200000        # 每 batch 最大 token 数
  required_batch_size_multiple: 1

distributed_training:
  distributed_world_size: 1 # 单 GPU 推理

decoding:
  beam: 1                   # beam=1 → Viterbi 解码
  type: viterbi             # 解码算法类型
```

### 5.2 命令行参数

实际运行时的完整命令：

```bash
python infer.py \
    --config-dir config \
    --config-name infer \
    task=spec_finetuning \                          # 使用微调任务类型
    task.data=/root/asr-competition/test_data \      # 数据目录
    task.normalize=false \                           # 不额外归一化
    common.user_dir=/root/asr-competition/ASR/data2vec_dialect \  # 自定义模块路径
    common_eval.path=/root/asr-competition/model_checkpoints/finetune_large_kespeech.pt \  # 模型权重
    common_eval.results_path=/root/asr-competition/decode_result \  # 结果输出
    common_eval.quiet=false \                        # 打印 HYPO/REF 对
    dataset.gen_subset=train \                       # 用 train 子集
    +task.target_dictionary=/root/asr-competition/model_checkpoints  # 字典目录
```

---

## 六、完整推理流程图

```
                        ┌──────────────────────┐
                        │   decode.sh 启动      │
                        │   (bash 脚本)          │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   hydra_main(cfg)     │
                        │   解析 YAML + 命令行   │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ InferenceProcessor()  │
                        │ 1. 加载模型权重        │
                        │ 2. 加载字典            │
                        │ 3. 初始化计数器         │
                        └──────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
    │  Batch 1 (N句)   │  │  Batch 2 (N句) │  │  Batch M (N句)  │
    │  feats: B×T×40  │  │  ...           │  │  ...            │
    │  targets: B×len  │  │                │  │                 │
    └─────────┬────────┘  └───────┬────────┘  └────────┬────────┘
              │                    │                    │
    ┌─────────▼────────────────────▼────────────────────▼────────┐
    │                    Wav2VecCtc.forward()                     │
    │  feats → ConvEncoder → Transformer → Linear(768→7531)      │
    │  输出: logits (B, T/4, 7531)                                │
    └────────────────────────┬───────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     CTC Viterbi Decode      │
              │  logits → argmax → 去重      │
              │  → 去 blank → ID → 文字      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    计算编辑距离 (CER/WER)     │
              │  editdistance(预测, 参考)     │
              │  errors += 当前错误数          │
              │  length += 参考总字数          │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     最终输出                  │
              │  WER = errors / length × 100│
              │  输出 HYPO/REF 对             │
              │  生成 wer 文件                │
              └─────────────────────────────┘
```

---

## 七、实际运行结果

```
===================================================
  finetune_large_kespeech.pt 推理结果 (35,793句)
===================================================
  CER (字错率):  7.03%  (36698 / 522086)
  SER (句错率):  43.15%  (15443 / 35793)
  处理速度:      97.6 句/秒, 366.6 秒
  GPU 显存:      ~20GB
===================================================
```

---

## 八、关键文件索引

| 文件 | 路径 | 功能 |
|------|------|------|
| **推理入口** | `ASR/data2vec_dialect/infer.py` | 主推理脚本，518 行 |
| **数据加载** | `ASR/data2vec_dialect/data/spec_dataset.py` | 读取 MFCC ark 文件 |
| **任务定义** | `ASR/data2vec_dialect/tasks/audio_finetuning.py` | spec_finetuning 任务 |
| **模型定义** | `fairseq/fairseq/models/wav2vec/wav2vec2_asr.py` | Wav2VecCtc + Wav2VecEncoder |
| **解码脚本** | `ASR/data2vec_dialect/run_scripts/decode.sh` | 启动推理的 bash 脚本 |
| **推理配置** | `ASR/data2vec_dialect/config/infer.yaml` | 推理参数 YAML |
| **微调配置** | `ASR/data2vec_dialect/config/v2_dialect_asr/base_audio_finetune_140h.yaml` | 微调参数 |
| **数据列表** | `test_data/data.list` | 核心数据索引文件 |
| **特征文件** | `test_data/feat/mfcc.*.ark` | 预提取 MFCC 特征 |
| **字典** | `model_checkpoints/dict.chr7531.txt` | 7531 个中文字符映射 |

---

## 九、环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA A100-SXM4-80GB |
| Python | 3.8.20 (conda: chuanyu-ASR) |
| PyTorch | 1.13.0+cu117 |
| Fairseq | 0.12.2 |
| CUDA | 13.0 |
| 服务器 | xj-member.bitahub.com:42066 |
