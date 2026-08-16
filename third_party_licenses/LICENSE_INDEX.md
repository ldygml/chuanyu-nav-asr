# 第三方许可证清单

> 本清单覆盖 submission 包（wheels/ 20 个依赖包 + 框架/参考代码）涉及的全部第三方软件。
> 完整许可证文本：核心框架见本目录 LICENSE_fairseq / LICENSE_wenet；
> wheels 内各包自带 dist-info 中的 LICENSE/METADATA，可在解包后核验。

## 一、框架与参考代码

| 软件 | 版本 | 许可证 | 来源 | 用途 |
|------|------|------|------|------|
| TeleSpeech-ASR | 1.0-large-kespeech | TeleSpeech 模型社区许可协议（见 LICENSE_TeleSpeech.pdf） | https://github.com/Tele-AI/TeleSpeech-ASR | 基座模型；本项目 src/ 派生自其 data2vec_dialect |
| fairseq | 0.12.2 | MIT | https://github.com/facebookresearch/fairseq | 训练/推理框架 |
| wenet | —（参考实现） | Apache-2.0 | https://github.com/wenet-e2e/wenet | 训练基线参考（未打包进推理代码） |

## 二、Python 依赖（wheels/）

| 包 | 版本 | 许可证 |
|------|------|------|
| torch | 2.6.0 | BSD-3-Clause |
| torchaudio | 2.6.0 | BSD-3-Clause |
| torchvision | 0.21.0 | BSD-3-Clause |
| numpy | 2.2.6 | BSD-3-Clause |
| kaldiio | 2.18.1 | Apache-2.0 |
| soundfile | 0.14.0 | BSD-3-Clause |
| sentencepiece | 0.2.0 | Apache-2.0 |
| editdistance | 0.8.1 | MIT |
| hydra-core | 1.3.5 | MIT |
| omegaconf | 2.3.1 | BSD-3-Clause |
| pyyaml | 6.0.3 | MIT |
| timm | 1.0.28 | Apache-2.0 |
| tensorboardX | 2.6.5 | MIT |
| Pillow | 12.2.0 | HPND (PIL 开源许可) |
| protobuf | 7.35.1 | BSD-3-Clause |
| antlr4-python3-runtime | 4.13.2 | BSD-3-Clause |
| packaging | 26.3 | BSD-2-Clause |
| typing_extensions | 4.16.0 | PSF |
| cffi | 2.1.1 | MIT |
| pycparser | 3.0 | BSD-3-Clause |

## 三、外部模型与数据（随包不打包，申报见 submission.yaml）

| 资源 | 许可证 | 说明 |
|------|------|------|
| TeleSpeech-ASR1.0-large-kespeech | TeleSpeech 模型社区许可协议（LICENSE_TeleSpeech.pdf） | 基座模型（仅限非商业使用；商用需向许可方登记并获书面授权） |
| nav80k_cont（本项目模型权重） | TeleSpeech 模型社区许可协议（同上） | 对 TeleSpeech-ASR1.0-large-kespeech 微调续训的模型衍生品，受同一协议约束 |
| KeSpeech phase1 子集 | 开源 | 通用预训练数据 |
| SoX 14.4 | LGPL | 离线数据增强 |
