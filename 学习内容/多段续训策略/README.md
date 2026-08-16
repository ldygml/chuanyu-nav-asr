# 多段续训策略（Multi-Stage Fine-tuning）

## 一、一句话概括

不是一次训到底，而是**分多段训练**，每段从上一段最优 checkpoint 重新微调。

---

## 二、为什么有效

```
单次长训练: 基线 ────────────→ 40k步 (可能陷在局部最优)
多段续训:   基线 → 40k最优 → 60k最优 → 80k最优
              ↓        ↓         ↓
            reset    reset     reset
```

每段 reset optimizer（重置优化器状态），让模型从当前最优出发，用新鲜动量重新探索，跳出局部最优。

---

## 三、本项目的三段续训

### 配置

```bash
# 每段通用配置
model.w2v_path=kespeech_encoder.pt
checkpoint.finetune_from_model=上段最优checkpoint
optimization.lr=[1e-06]  # 极低LR，保留已有知识
```

### 效果

| 阶段 | 步数 | 起点 | 导航 CER |
|------|:--:|------|:--:|
| 1 | 0→40k | finetune_large_kespeech | 4.93% |
| 2 | 0→20k | 40k最优 | 1.88% |
| 3 | 0→20k | 60k最优 | **0.70%** |

### 关键参数

```
finetune_from_model = 加载完整模型权重（含CTC头）
optimizer = 重新初始化 Adam
lr_scheduler = 从头开始 tri_stage
learning_rate = 1e-06 (始终保持极低)
```

### 为什么 LR 要极低

- 模型已经很好（4.93% CER），大 LR 会破坏已有知识
- 1e-06 只能做微小的方向调整
- 配合 reset optimizer，相当于"从最优位置重新搜一遍"

---

## 四、和普通续训的区别

| | 普通续训 | 多段续训 |
|------|------|------|
| optimizer | 保留 | **重置** |
| LR scheduler | 继续 | **从头** |
| 起点 | 最后一帧 | **最优帧** |
| 效果 | 可能过拟合 | 每段提升 |

---

## 五、注意

- 仅适用于**已接近最优**的模型（CER < 5%）
- 如果模型还很差（CER > 20%），reset optimizer 会丢失进度
- 建议每段 10k-20k 步，总步数不宜超过初始训练的 2 倍
