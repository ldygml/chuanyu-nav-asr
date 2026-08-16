# 决赛材料总览

**队伍**: LynRose Enigma | **赛项**: CCF IVC 2026 智能导航方言语音识别挑战赛
**版本**: v2.0（nav80k_cont 模型）| 更新: 2026-08-13

## 材料索引（对应细则 2.3 / 3.2 要求）

| 要求 | 材料 | 位置 |
|------|------|------|
| 决赛提交包（3.2/3.3） | run.sh + src + weights + fairseq + wheels + yaml | `submission/` |
| 技术说明 | 模型简介/训练流程/推理设置/随机性控制/包大小 | `submission/README.md` |
| 提交元数据 | 权重清单 + SHA-256 + 外部资源申报 | `submission/submission.yaml` |
| 完整项目源代码 | 推理代码 + 训练脚本 | `submission/src/`、`训练脚本/` |
| 开源数据集列表 | 2.3 a) | `开源数据集列表.md` |
| 自用数据集制作流程 | 2.3 a) | `自用数据集制作流程.md` |
| 环境配置文档 | 2.3 a) | `环境配置文档.md` |
| 赛事方案文档 | 2.3 b) 模型思路/优化/实验/创新点 | `赛事方案文档.md` |
| 成果说明材料 | 2.3 c) 成果清单+合规开源声明 | `成果说明材料.md` |
| 第三方许可证清单 | 3.2 | `submission/third_party_licenses/LICENSE_INDEX.md` |

## 打包提交

```bash
# 决赛提交包（tar.gz，<10GB）
tar -czf submission_final.tar.gz submission/
```

当前权重为 nav80k_cont（3.8GB），归档后约 4.3GB。
