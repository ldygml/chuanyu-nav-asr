# 比赛评测体系

## 一、官方工具

比赛方提供的两个 Python 脚本：

| 工具 | 功能 |
|------|------|
| `normalize.py` | 文本规范化：去标点、只保留中文字母数字 |
| `score.py` | CER 计算：字符级编辑距离 |

---

## 二、normalize.py

### 功能

删除所有非中文字符、非英文字母、非数字的字符：
- 标点符号：，。！？、；：""''（）
- 空格
- 特殊符号：@#$%^&*

### 示例

```
原始: 我明天上午9点,要从什邡市方亭街道办出发！
规范: 我明天上午9点要从什邡市方亭街道办出发
```

### 两种模式

```bash
# 处理原始 TSV 标注
python normalize.py -m tsv -i UTTERANCEINFO.txt -o ref.jsonl

# 处理模型预测结果
python normalize.py -m jsonl -i model_output.jsonl -o predictions.jsonl
```

---

## 三、score.py

### CER 公式

```
CER = (S + D + I) / N

S = 替换次数 (Substitution)
D = 删除次数 (Deletion)
I = 插入次数 (Insertion)
N = 参考文本总字符数
```

### 用法

```bash
python score.py -r reference.jsonl -t predictions.jsonl -o result.jsonl
```

### 输出

```
平均 CER: 0.0070  (共 3385 条)
结果已写入: result.jsonl
```

### 与原始 CER 的区别

| 方法 | v8 CER |
|------|:--:|
| editdistance 原始 | 5.90% |
| normalize + score | **5.50%** |

规范化去标点后 CER 降约 0.4pp。

---

## 四、提交流程

```bash
# 1. 推理
python infer.py \
    --config-name infer \
    task.data=/path/to/test_data \
    common_eval.path=/Save/checkpoints_nav80k/checkpoint_best.pt \
    common_eval.quiet=false

# 2. 提取 HYPO → JSONL
python3 -c "
import json
hypos = []
with open('infer.log') as f:
    for l in f:
        if 'HYPO:' in l: hypos.append(l.split('HYPO: ',1)[1].strip())
with open('predictions_raw.jsonl', 'w') as f:
    for i, h in enumerate(hypos):
        f.write(json.dumps({'id': f'{i:04d}', 'text': h}, ensure_ascii=False) + '\n')
"

# 3. 规范化
python normalize.py -m jsonl -i predictions_raw.jsonl -o predictions.jsonl

# 4. 提交 predictions.jsonl
```

---

## 五、本地自测流程

```bash
# 1. 生成参考文本
python normalize.py -m tsv -i UTTERANCEINFO.txt -o ref.jsonl

# 2. 模型推理 → predictions_raw.jsonl

# 3. 规范化预测
python normalize.py -m jsonl -i predictions_raw.jsonl -o pred.jsonl

# 4. 评分
python score.py -r ref.jsonl -t pred.jsonl -o result.jsonl
```
