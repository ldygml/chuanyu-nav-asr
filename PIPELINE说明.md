# PIPELINE 完整说明（上下文恢复文档）

> 用途：上下文压缩后快速恢复全部状态。**后续只会改动热词后处理部分**（第 3 节），ASR 部分（第 2 节）视为冻结。
> 更新：2026-08-13 18:55

---

## 1. 项目概况

- 比赛：CCF IVC 2026 智能导航方言语音识别挑战赛（川渝方言导航语音），赛项三
- 阶段：初赛 A 榜，截止 **2026-08-14**（提交 predictions.jsonl 即可）；决赛 8/19-20（提交模型封闭评测）
- 队伍：LynRose Enigma
- 基线：nav80k_cont 模型 greedy 解码，A 榜 CER = **6.68%**
- 提交次数限制：A 榜每日 ≤3 次，全程 ≤20 次
- 用户工作模式：训练/推理类重活一律在服务器跑（"你别在本地跑"）；热词脚本可本地跑

### 关键文件地图

| 位置 | 路径 |
|---|---|
| 工作目录（本地） | e:\lg |
| 提交目录（本地） | e:\lg\初赛提交\ |
| 初赛测试集（本地） | e:\lg\初赛测试集\（TestDataset/TestDataset/TEST/*.wav，3461 条） |
| 决赛材料（本地） | e:\lg\决赛材料\（submission 包 + 文档 + 训练脚本） |
| 热词库（本地） | e:\lg\asr-hotword\（HaujetZhao，从 CapsWriter-Offline 抽离） |
| 服务器持久盘 | /Save（重启不丢） |
| 服务器易失区 | /tmp（**服务器重启即丢**，热词三件套在此，丢了要重传） |

---

## 2. ASR 部分（冻结，勿改）

### 2.1 服务器连接

```bash
ssh -i E:/lg/id_rsa -p <端口> root@xj-member.bitahub.com
scp -i E:/lg/id_rsa -P <端口> <本地> root@xj-member.bitahub.com:<远程>
```
- **端口会变**：当前 42096（2026-08-15 起），此前用过 42112、42104、42025、42169。换端口后 host key 验证会失败 → 先 `ssh-keyscan -p <端口> xj-member.bitahub.com >> ~/.ssh/known_hosts`
- Python：`/opt/conda/envs/chuanyu-ASR/bin/python3`（有 torch/kaldiio/pypinyin/rapidfuzz）
- GPU：A100 80GB

### 2.2 模型

- 提交模型：`/Save/checkpoints_nav80k_cont/checkpoint_best.pt`（3.8GB）
- 训练路线：nav80k（导航增强 80k 步）→ nav80k_cont（续训 20k 步，LR=1e-06，2026-08-12 完成）
- 训练命令已归档：本地 `决赛材料/训练脚本/nav80k_cont续训.sh`（从 train.log cfg 重建）
- 词典：`/root/asr-competition/submission/weights/dict.chr7531.txt`（7531 字）
- 其他尝试过的模型（勿用）：navmix（M_predict 4.14%，略好但不稳定）、nav_specaug（已失败 8.64%）、FC1/FC2（已放弃并全部删除）

### 2.3 推理流程（官方三件套）

```
wav → prepare_manifest_data.py (40维MFCC+data.list)
    → infer.py (hydra, task=spec_finetuning, decoding=viterbi greedy)
    → make_predictions.py (按 IDX 排序对齐 + 去<unk> → jsonl)
```

- 组件位置：`/root/asr-competition/submission/src/`（服务器）；`e:\lg\决赛材料\submission\src\`（本地）
- 测试数据：wav 在 `/Save/TEST/*.wav`（3461），manifest 在 `/Save/初赛_test_manifest.jsonl`
- **一体化脚本**：`/root/asr-competition/run_pipeline.sh`（4 步：MFCC→推理→中间jsonl→**热词修正**→输出；参数 `--input_manifest/--output_path/--device`）
- 完整推理产物（无热词）：`/Save/predictions_nav80k_cont_greedy_v2.jsonl` = 本地 `初赛提交/predictions_nav80k_cont_greedy_v2.jsonl`（已验证与 8/12 旧版 3461/3461 逐字一致）
- ★**通用转写工具（2026-08-16 重写版 pipeline）**：`/root/asr-competition/transcribe.sh` = 本地 `e:\lg\transcribe.sh`
  ```
  bash transcribe.sh --input <wav文件 或 目录> [--output out.jsonl] [--no-hotword] [--device cuda:0]
  ```
  任意采样率/声道自动重采样 16k 单声道；免写 manifest（id=文件名）；stdout 打印 `id<TAB>text`；热词默认开启（557 对），`--no-hotword` 跳过。已实测：目录模式、单文件模式、44.1kHz 重采样、热词开关均通过（dev 2 条与 ref 逐字一致）。底层三件套未改动，只是新包了一层入口

### 2.4 ASR 复现注意事项

- 决赛权重 weights/model.pt 曾在服务器被换入换出（备份 /tmp/model.pt.final.bak），现已恢复
- run.sh 会 `rm -rf` 自己的临时目录，产物直接写 --output_path
- 推理 3461 句耗时：MFCC ~20 分钟（CPU），greedy ~90 秒（GPU）

---

## 3. 热词后处理部分（★后续唯一改动区）

### 3.1 三件套

| 件 | 本地路径 | 服务器路径（/tmp，易失） |
|---|---|---|
| 脚本 | e:\lg\hotword_fix.py | /tmp/hotword_fix.py |
| 词典 | e:\lg\初赛提交\hotword_dict_final.md（★当前权威，557 对） | /tmp/hotword_dict_final.md |
| 库 | e:\lg\asr-hotword\ | /tmp/asr-hotword\（新脚本纯字典替换，已不依赖） |

- ★2026-08-14 词典换代：hotword_dict_final.md 来自用户新找的「川渝方言导航热词映射(1).md」主表（高置信）479 对，**用户确认无违规**（旧 human_correct.md 测试集派生、已弃用；川渝地名热词.md 未并入本次提交版）。该文件原始格式为 `错词 → 对词`，按 ' → ' 分隔解析；低置信段 82 对已排除（用户选择仅主表）
- ★2026-08-15 并入「已验证」段：源文件 8/14 18:26 更新后，旧低置信 82 条被拆分重组为「已验证」79 条（用户人工/音频核验）+「待核验」28 条。已验证 79 条已全部并入词典，苏州高苏区用户确认用姑苏区（高新区条目已删）→ **557 对** = 主表 479 + 已验证 78 + 主表外滩条目修正。服务器词典文件已改名 /tmp/hotword_dict_final.md（原 human_correct.md 已删，脚本 DICT 同步改）
- ★2026-08-15 v5：用户要求从语音全流程重跑（run_pipeline.sh：MFCC→greedy→热词），452 句修改。新鲜 ASR 输出与 v2 逐字一致（推断确定性复现）；v5 vs v4 差异 62 句（纯词典变化导致）
- 旧词典历史：merged 版 1522 对（human_correct 980 + 地名热词 500 + 编辑），产物 v3
- 2026-08-14 v4 运行：400 句修改（vs v2），478 kept / 30 reverted（全为子串防护/PREV_EXCEPT 拦截）

- 服务器版脚本与本地逻辑完全相同，仅两处路径适配：`DICT='/tmp/hotword_dict_final.md'`、`sys.path.insert(0,'/tmp/asr-hotword')`
- 词典格式：每行 `错误词 正确词`（空格分隔），当前 557 行；用户会持续编辑
- 依赖：pypinyin + rapidfuzz（服务器 chuanyu-ASR 环境已装，`deps ok` 验证过）

### 3.2 当前算法逻辑（hotword_fix.py）

1. 加载词典对（错→对），构建 `wrong_set`；`PhonemeCorrector(threshold=0.85)` 载入热词（格式 `正确词 | 错误词`）
2. 逐句：去标点归一 → `pc.correct()` 得匹配列表（matches 按位置 **DESC** 排序返回 (原词,热词,分数)）
3. **词库精确判据**：仅当 wrong 侧**精确等于**词典条目才替换；音素变体一律恢复（reverted）
4. **子串防护**：若 `len(r) > len(w)` 且 w 左右邻接汉字 → 恢复（拦"没耽→没耽搁"顶出"没耽误"的"误"字）
5. **手动例外 REVERT_SET**：`{('久曲路','旧曲路'), ('经点','景点')}` —— 用户点名/dev 实证的伤害对
6. 替换用 `rfind(w, 0, last_end)` 游标从右往左，防止替换产物中的子串被二次替换
7. 输出统计：changed 句数、kept/reverted 数、Top15 对

### 3.3 更新词典后的标准操作流程

```bash
# ① 用户编辑本地词典后，上传（端口按当前）
scp -i E:/lg/id_rsa -P 42096 e:/lg/初赛提交/hotword_dict_final.md root@xj-member.bitahub.com:/tmp/hotword_dict_final.md

# ② 若 /tmp 丢了热词脚本（服务器重启过），一并重传
scp -i E:/lg/id_rsa -P 42096 e:/lg/hotword_fix.py root@xj-member.bitahub.com:/tmp/hotword_fix.py
# （asr-hotword 库丢了就 scp -r 整个目录）

# ③ 服务器对 ASR 结果跑热词
ssh -i E:/lg/id_rsa -p 42096 root@xj-member.bitahub.com \
  "/opt/conda/envs/chuanyu-ASR/bin/python3 -X utf8 /tmp/hotword_fix.py \
   /Save/predictions_nav80k_cont_greedy_v2.jsonl /Save/predictions_hotword_new.jsonl"

# ④ 下载回本地（初赛提交/）
scp -i E:/lg/id_rsa -P 42096 root@xj-member.bitahub.com:/Save/predictions_hotword_new.jsonl e:/lg/初赛提交/

# ⑤ 本地快速核验（BOM/行数/id 集合/字段/规范化/空文本）
python -X utf8 - <<'EOF'
import json, re, glob, os
lines = open(r'e:/lg/初赛提交/predictions_hotword_new.jsonl', encoding='utf-8').read().splitlines()
assert len(lines) == 3461, len(lines)
ids = set()
for i, l in enumerate(lines):
    o = json.loads(l)
    assert set(o) == {'id', 'text'}, (i, o.keys())
    assert o['id'] not in ids; ids.add(o['id'])
    assert o['text'] == re.sub(r'[^一-鿿A-Za-z0-9]', '', o['text']), (o['id'], o['text'])
wav_ids = {os.path.basename(p)[:-4] for p in glob.glob(r'e:/lg/初赛测试集/TestDataset/TestDataset/TEST/*.wav')}
assert ids == wav_ids, (len(ids ^ wav_ids))
print('OK', len(lines), 'no-BOM:', open(r'e:/lg/初赛提交/predictions_hotword_new.jsonl','rb').read(3) != b'\xef\xbb\xbf')
EOF
```

### 3.4 质量验证手段

- **dev 105 有真值**：`e:\lg\初赛测试集\_cer_result.jsonl`（{id,text,ref,cer}），预测在 `_pred_n80kcont_105.jsonl`
- dev 105 CER 对比（纯 Python Levenshtein，不用 editdistance——用户拒绝 pip install）：
  - baseline（无热词）：**2.3655%**
  - 历史：音素全替换 4.96%（坏）→ 词库精确版 3.34%（2102行词典）→ **合并词典版 3.2468%**（2026-08-13，1479行）
  - ⚠️ 2026-08-13 关键发现：dev 105 上热词修改 34 句 = 变好 1 / 变差 26 / 持平 7。变差原因：**官方 ref 是按音转写**（ref 写"华音山""环尔赛""苏国超市""京开大道""要家上"等音似错字），热词改成规范地名（华蓥山/凡尔赛/苏果/金开/要加上）反而与 ref 偏离。若 A 榜 ref 同风格，热词版会伤分——基线 6.68% 是纯模型成绩，提交热词版前需用户权衡
- **误伤回归检查**（改词典后必查三个模式在输出中的出现次数，应为 0）：
  `没耽搁误`、`自驾车去`、`旧曲路`（另 `途景点` 允许 1 处="沿途景点"原文固有）
- 上次全量结果（2102 行词典）：686 句修改，812 kept / 2042 reverted；Top kept 途径点→途经点(16)

### 3.5 热词产物

| 文件 | 说明 |
|---|---|
| 本地 初赛提交/predictions_nav80k_cont_greedy_v5.jsonl | **★当前最新热词版**（8/15，从语音全流程重跑，557 对词典，452 句修改，已全量核验） |
| 本地 初赛提交/predictions_nav80k_cont_greedy_v4.jsonl | 8/14 版（final 词典 479 对，400 句修改；A 榜提交版 = 此版） |
| 本地 初赛提交/predictions.jsonl | **★A 榜提交文件** = v4 副本（8/14 生成） |
| 本地 初赛提交/predictions_nav80k_cont_greedy_v3.jsonl | 上一版（merged 1522 对词典，已过时） |
| 服务器 /Save/predictions_nav80k_cont_greedy_v4.jsonl | v4 服务器原件 |
| 服务器 /Save/predictions_pipeline_full.jsonl | pipeline 一体化输出（与分步结果 3461/3461 一致） |

---

## 4. 合规红线（重要，提交前必读）

细则 2.3 / 7.2 明文禁止：
1. **不得使用由测试集直接衍生的任何数据、模型、词典与规则**
2. 测试音频不得用于**人工转录、人工校对、构建答案映射表**

✅ 2026-08-14 用户声明：**所有热词整理（human_correct.md、川渝地名热词.md、川渝方言导航热词映射(1).md）均来自测试集以外的资源**（外部地名/机构名资料、常识整理），与测试集无关 → 热词版提交合规，无违规风险。此前的"测试集派生风险"判断已由用户澄清撤销。

提交文件规范：UTF-8 无 BOM、3461 行、仅 id/text 字段、id 与 TEST 目录一致、文本符合官方 normalize 规则（只保留汉字/英文/数字）。

---

## 5. 材料状态（决赛，已按细则整理完）

| 材料 | 状态 |
|---|---|
| e:\lg\决赛材料\submission\ | 完整（run.sh/README/yaml/src/weights/fairseq/wheels/licenses） |
| weights/model.pt | 已是 nav80k_cont（SHA-256: fa8ce92c...，md5: 40d38f0c...，与服务器一致） |
| submission.yaml | v2.0，80k_cont 描述 + SHA-256 已填 |
| 成果说明材料.md / 开源数据集列表.md / 自用数据集制作流程.md / 环境配置文档.md / 赛事方案文档.md | 齐全 |
| 训练脚本/ | 18 个脚本归档 + nav80k_cont续训.sh |
| submission_final.tar.gz | **仍是 8/10 旧版（v1.1 权重）**，提交前需重新打包 |

---

## 6. 踩坑记录（防重犯）

1. **服务器 /tmp 易失**：热词三件套在 /tmp，服务器重启后要重传（词典+脚本+asr-hotword 库）
2. **端口会变**：每次 SSH 前确认当前端口
3. **pkill -f 自杀**：pkill -f 'run_xxx.sh' 会匹配到自己 SSH 会话的命令行（含目标字符串）→ exit 255；杀进程先 `ps aux` 核完整命令行，优先 kill 精确 PID
4. **被拒的 heredoc 可能已执行**：一次被用户拒绝的脚本上传实际在服务器跑起来了，与后续流程形成双进程竞争同一输出文件；启动前先 `ps aux | grep` 确认没有同名进程
5. **Windows Git Bash 中文乱码**：跑 Python 加 `-X utf8`，输出写 UTF-8 文件后用 Read 工具看，别依赖控制台
6. **用户拒绝 pip install**（editdistance）：评测用纯 Python Levenshtein（见 hotword_dev_eval.py / hotword_pair_analysis.py）
7. **DataLoader worker 误杀**：查 PPID 再 kill（fork 继承的命令行与主进程相同）
8. **用户偏好复用现成物**：优先用服务器已有脚本（run.sh、run_pipeline.sh），不要动辄新传脚本

---

## 7. 后续待办

- [ ] A 榜提交（8/14 前）：当前 predictions.jsonl = 纯模型版；若用户改热词后再定
- [ ] 决赛打包：submission_final.tar.gz 用 80k_cont 权重重新打包
- [x] dev 105 复测合并词典版：CER 3.2468%（baseline 2.3655%，热词仍伤 dev；见 3.4 警告）
- [x] 8/14 final 词典版 v4：dev 105 = 2.6438%（479 对收敛后伤分减小，仍高于 baseline 2.3655%）
