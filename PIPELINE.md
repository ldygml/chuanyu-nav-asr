# PIPELINE 说明

> 用途：介绍本项目的完整推理流水线、训练路线与热词后处理。
> 更新：2026-08-16

---

## 1. 项目概况

- 比赛：CCF IVC 2026 智能导航方言语音识别挑战赛（川渝方言导航语音），赛项三
- 队伍：LynRose Enigma
- 成绩：初赛第 14 名（A 榜纯模型基线 CER = 6.68%，热词后处理版提交）
- 模型：Data2Vec 2.0 编码器 + CTC，checkpoint = nav80k_cont

## 2. 训练路线

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

- 训练命令见 `scripts/`（含 nav80k_cont 续训脚本）
- 失败的探索（勿复现）：navmix（混合通用数据续训）、nav_specaug（加噪 + SpecAugment）、FastCorrect 纠错模型微调

## 3. 推理流程

### 3.1 底层三件套

```
wav → prepare_manifest_data.py（任意采样率自动重采样 16k 单声道 → 40 维 MFCC + data.list）
    → infer.py（hydra，task=spec_finetuning，viterbi greedy 解码）
    → make_predictions.py（按 IDX 排序、去 <unk> → jsonl）
```

### 3.2 通用转写工具（推荐入口）

```bash
bash transcribe.sh --input <wav文件 或 目录> [--output out.jsonl] [--no-hotword] [--device cuda:0]
```

- 免写 manifest（id = 文件名）；stdout 打印 `id<TAB>text`
- 热词修正默认开启（`hotword/hotword_dict_final.md`），`--no-hotword` 跳过
- 环境变量：`NAV_ASR_CKPT`（权重路径，默认 `weights/model.pt`）、`NAV_ASR_PYTHON`（解释器）、`HOTWORD_DICT`（词典路径）
- 权重：Hugging Face `999ffgml/chuanyu-nav-asr`（`hf download 999ffgml/chuanyu-nav-asr --local-dir weights`）

## 4. 热词后处理（hotword_fix.py）

### 4.1 算法逻辑

1. 加载词典对（错→对），按**错词长度降序**排序（避免嵌套对重复套用）
2. 逐句去标点归一后 find 替换，游标递进（替换产物不会被二次替换）
3. **子串防护**：加长替换（`len(对) > len(错)`）且左右邻接汉字 → 该处跳过
4. **手动例外 REVERT_SET**：`{('久曲路','旧曲路'), ('经点','景点')}` —— dev 实证的伤害对，整对跳过
5. **上下文例外 PREV_EXCEPT**：`{('约后','约好'): '预'}` —— "预约后天" 里的"约后"不替换
6. 输出统计：changed 句数、kept/reverted 次数、Top15 对

### 4.2 词典

- 权威词典：`hotword/hotword_dict_final.md`（**1981 对**，2026-08-16 合并去重版）
  - 来源：`川渝方言导航热词映射(1).md`（主表 + 已验证，新表优先）∪ `hotword_dict_merged.md`（历史词库）
  - 去重规则：同错词冲突以新表为准（6 处）；苏州高苏区仅保留"姑苏区"目标
- 格式：每行 `错误词 正确词`（空格分隔），`#` 开头为注释
- 词典全部来自测试集以外的外部地名/机构名资料与常识整理
- 迭代历程：音素匹配版（asr-hotword）→ 纯字典替换版（2102 行，dev CER 3.34%）→ 子串防护 + 收敛词典（557 对，dev CER 2.64%）→ 合并去重版（1981 对）

### 4.3 质量验证

- dev 105 有官方真值，纯 Python Levenshtein 评测（`editdistance` 可替）
- baseline（无热词）dev CER = 2.3655%；热词版 2.6438% —— 官方 ref 按音转写（如"华音山""环尔赛"），热词改成规范地名（华蓥山/凡尔赛）会与 ref 偏离；A 榜 ref 若同风格需权衡
- **误伤回归检查**（改词典后必查输出中出现次数应为 0）：`没耽搁误`、`自驾车去`、`旧曲路`

## 5. 可复现性

- 推理确定性：固定 seed=777；两次独立推理 3461/3461 逐字一致
- 权重 SHA-256 见 `submission.yaml`；输入任意采样率/声道数的 WAV 均可
- 推理耗时（3461 句，A100）：MFCC ~20 分钟（CPU），greedy ~90 秒（GPU）
