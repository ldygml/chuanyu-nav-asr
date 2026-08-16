# Fairseq Hydra 训练启动机制详解

> 学习日期: 2026-08-10 | 来源: 实战中踩坑总结

---

## 一、一句话概括

fairseq 训练用 Hydra config 管理参数。启动时 Hydra 先解析 override → struct mode 校验 → 加载用户模块 → `add_defaults` 补全 schema。**必须用含 `_name` 字段的 config YAML，不能用仅有 `null` 的默认 config。**

---

## 二、整体流程

```
1. hydra_init()
   ├── ConfigStore 注册 FairseqConfig schema
   └── 注册每个子字段 (model, task, optimizer, ...) 的默认值

2. @hydra.main 装饰器
   ├── 解析命令行 override
   ├── Struct mode 校验: 覆盖的 key 必须在 schema 中
   ├── 加载 config YAML + defaults list
   └── 调用 hydra_main(cfg)

3. _hydra_main(cfg)
   ├── add_defaults(cfg)  ← 关键！
   │   ├── OmegaConf.set_struct(cfg, False)  # 暂时关闭 struct mode
   │   ├── 遍历 FairseqConfig 字段
   │   │   ├── 如果 field_cfg 是 str → 转为 DictConfig({_name: str})
   │   │   ├── 查 MODEL_DATACLASS_REGISTRY / TASK_DATACLASS_REGISTRY
   │   │   └── merge_with_parent: 把 dataclass 的完整 schema 合并进来
   │   └── OmegaConf.set_struct(cfg, True)   # 重新开启
   ├── 训练
```

---

## 三、两个 config 源

### fairseq 默认 config.yaml
路径: `fairseq/config/config.yaml`
```yaml
defaults:
    - model: null      # ← 问题根源！
    - task: null
    - optimizer: null
    - lr_scheduler: fixed
```

**问题**: `model: null` 意味着 struct mode 下无法 override `model.w2v_path`

### 用户模块 config YAML (推荐)
路径: `data2vec_dialect/config/v2_dialect_asr/base_audio_finetune_140h.yaml`
```yaml
model:
  _name: wav2vec_ctc    # ← 预填 schema name
  w2v_path: ???
task:
  _name: spec_finetuning
optimizer:
  _name: adam
lr_scheduler:
  _name: tri_stage
```

**优点**: 预填了 `_name` → struct mode 知道 schema → 可以 override 子字段

---

## 四、关键函数

### `hydra_init()` — ConfigStore 注册
```python
def hydra_init(cfg_name="config") -> None:
    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=FairseqConfig)
    # 把 model=Any, task=Any, ... 存为 config group
    for k in FairseqConfig.__dataclass_fields__:
        v = FairseqConfig.__dataclass_fields__[k].default
        cs.store(name=k, node=v)
```

### `add_defaults()` — Schema 补全
```python
def add_defaults(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)  # 先关 struct mode
    for k, v in FairseqConfig.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any:
            # 从 _name 解析实际 dataclass
            name = getattr(field_cfg, "_name", None)
            if k == "model":
                name = ARCH_MODEL_NAME_REGISTRY.get(name, name)
                dc = MODEL_DATACLASS_REGISTRY.get(name)
            elif k == "task":
                dc = TASK_DATACLASS_REGISTRY.get(name)
            # 合并完整 schema
            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)
    OmegaConf.set_struct(cfg, True)
```

**关键点**: `add_defaults` 在 struct mode 关闭时运行 → 可以自由修改 config → 然后重新开启 struct mode

---

## 五、两种加载方式

### `finetune_from_model`
```
用途: 新训练，从预训练模型起步
行为:
  ├── 加载 checkpoint 的 model 权重
  ├── 初始化新 optimizer（随机）
  ├── 初始化新 lr_scheduler（从头）
  └── ❌ 不能同时设 reset_*=true（互斥）
```

### `restore_file`
```
用途: 续训，从断点恢复
行为:
  ├── 加载 checkpoint 的完整状态
  ├── 恢复 optimizer momentum
  ├── 恢复 lr_scheduler 状态
  └── ✅ 可以设 reset_*=true（需要时可以重置）
```

### 多段续训策略用的就是 `finetune_from_model`

```
段1: finetune_from_model=kespeech_encoder.pt  → 40k步 → 最优checkpoint
段2: finetune_from_model=段1最优               → 20k步 → 最优checkpoint
段3: finetune_from_model=段2最优               → 20k步 → 最优checkpoint
```

每段都自动重置 optimizer 和 lr_scheduler，从最优位置重新搜索。

---

## 六、常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `Key 'xxx' is not in struct` | config 里对应节点为 null | 用有 `_name` 的 config YAML |
| `Could not load task/xxx` | 用户模块未在 ConfigStore 中 | 不用 `task=xxx` override，让 config YAML 的 `_name` 处理 |
| `reset_optimizer + finetune_from_model` | 互斥参数 | 去掉 reset_* |
| `FileNotFoundError: xxx.tsv` | 缺少 tsv 文件 | `ln -sf data_xxx.list xxx.tsv` |
| `Failed to load xxx.ark` | 数据路径不存在 | 软链接到实际位置 |

---

## 七、正确的启动命令模板

```bash
cd /root/asr-competition/ASR/data2vec_dialect
export PYTHONPATH=/root/asr-competition/fairseq:$PWD:$PYTHONPATH

python /root/asr-competition/fairseq/fairseq_cli/hydra_train.py \
    --config-dir config/v2_dialect_asr \          # YAML 所在目录
    --config-name base_audio_finetune_140h \       # YAML 文件名
    common.user_dir=/root/asr-competition/ASR/data2vec_dialect \
    model.w2v_path=/path/to/encoder.pt \           # 不带 +，已存在于 YAML schema
    task.data=/path/to/data \                      # 不带 +
    +task.target_dictionary=/path/to/dict \        # 带 +，YAML schema 中没有
    checkpoint.finetune_from_model=/path/to/best.pt \
    checkpoint.save_dir=/Save/new_checkpoint \
    optimization.max_update=40000 \
    'optimization.lr=[5e-06]'
```

**规则**: YAML 里已有的字段 → 不带 `+`；YAML 里没有的 → 带 `+`
