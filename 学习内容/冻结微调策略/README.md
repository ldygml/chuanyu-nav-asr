# 冻结微调策略（Freeze Fine-tuning）

## 一、一句话概括

训练初期**锁住预训练 Encoder 不更新**，只让 CTC 分类头学习"语音特征→文字"的映射。训一段时间后再放开 Encoder 做整体微调。

---

## 二、模型结构回顾

```
Wav2VecCtc（整个模型）
│
├── Encoder（预训练，懂语音）
│   ├── 卷积前端（MFCC → 高维特征）
│   ├── Transformer 12-16 层
│   └── 输出：每帧一个 768/1024 维向量
│
└── CTC 分类头（随机初始化，需要训练）
    └── Linear(768/1024 → 7531) → 每个中文字符的概率
```

| 组件 | 参数量 | 预训练 | 能力 |
|------|:--:|:--:|------|
| Encoder | ~300M | 300k 小时语音 | 听清每个音素 |
| CTC 头 | ~8M | 无 | 把语音特征映射到文字 |

---

## 三、冻结的执行方式

```python
# freeze_finetune_updates = 15000

for step in range(total_steps):
    if step < 15000:
        # 冻结阶段：只更新 CTC 头的参数
        for param in model.encoder.parameters():
            param.requires_grad = False  # 🔒 不计算梯度
        for param in model.ctc_head.parameters():
            param.requires_grad = True   # 🟢 正常更新
    else:
        # 解冻阶段：全部参数参与训练
        for param in model.parameters():
            param.requires_grad = True   # 🟢 全部更新
```

效果：冻结期只更新 ~8M 参数（CTC 头），解冻后更新全部 ~300M 参数。

---

## 四、为什么冻结有效

### 4.1 不冻结的问题

```
预训练 Encoder（懂 30 万小时通用语音）
    ↓ 直接用少量方言数据微调
    ↓ Encoder 每次反向传播都被拉扯
    ↓ 小数据量下容易"过拟合"训练集的特定模式
    ↓ 忘记了预训练学到的通用语音知识
    → 测试集泛化差（第一轮 CER 15.5%）
```

### 4.2 冻结的思路

```
预训练 Encoder（懂 30 万小时通用语音，包含方言）
    ↓ 🔒 锁住不动
    ↓ CTC 头先学"方言文本映射"这个相对简单的任务
    ↓ CTC 头学会后 → 🟢 放开 Encoder
    ↓ 小步微调 Encoder，不破坏已有的语音理解基础
    → 泛化更好（第四轮 CER 7.08%）
```

### 4.3 直观类比

| 场景 | 不用冻结 | 用冻结 |
|------|------|------|
| **学方言** | 让一个已经懂中文的人立刻用四川口音说话 | 先背四川话词汇表和文字，再微调口音 |
| **学画画** | 让画家立刻换画风，可能画不好 | 先学新风格的理论，再练习笔法 |
| **转行** | 工程师直接做销售 | 先培训产品知识，再上岗 |

---

## 五、冻结步数的选择

| 冻结步数 | 效果 | 说明 |
|:--:|------|------|
| 0（不冻结） | CER 7.53% | Encoder 被过早拉动 |
| 5,000 | CER 7.42% | 略有提升 |
| 10,000 | CER **7.08%** 🏆 | 当前最优 |
| 15,000 | 训练中 | 正在测试极限 |

选择原则：

```
冻结步数太少 → CTC 头还没学好 → Encoder 被拖累
冻结步数太多 → CTC 头过拟合 → Encoder 来不及微调
最优范围 → 约占总步数的 25%-50%
```

---

## 六、解冻后的训练

解冻不是"突然放开"，而是渐进式的效果：

```
冻结期（前 15k 步）
├── GPU 显存：~5GB（只有 CTC 头在训练）
├── 训练速度：~1.3 ups（更新参数少）
└── Loss 下降快（CTC 头从零开始学）

解冻后（15k-20k 步）
├── GPU 显存：~19GB（全部参数训练）
├── 训练速度：~0.46 ups（更新参数多）
└── Loss 缓慢下降（微调 Encoder）
```

---

## 七、代码实现

训练脚本中的关键参数：

```bash
python hydra_train.py \
    model.w2v_path=kespeech_encoder.pt \
    model.freeze_finetune_updates=15000 \  # 前 15000 步冻结
    optimization.lr='[5e-06]' \            # 解冻后用这个 LR
    ...
```

Fairseq 内部实现（`wav2vec2_asr.py`）：

```python
class Wav2VecEncoder(nn.Module):
    def set_num_updates(self, num_updates):
        # 根据 freeze_finetune_updates 控制梯度
        if num_updates < self.cfg.freeze_finetune_updates:
            # 冻结：encoder 参数不需要梯度
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            # 解冻
            for p in self.encoder.parameters():
                p.requires_grad = True
```

---

## 八、我们的实验结果

| 冻结步数 | 外部 CER | 变化 |
|:--:|------|:--:|
| 0 | 7.53% | 基准 |
| 5,000 | 7.42% | ↓0.11pp |
| 10,000 | **7.08%** | ↓0.45pp |
| 15,000 | 待测 | — |

冻结 10k 比不冻结好了 0.45 个百分点，是当前最优策略。
