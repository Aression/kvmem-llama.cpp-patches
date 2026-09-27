# 量化实测数据（RCO 配方）

Qwen3.8 系列 27B dense 模型在 AMD RX 9070 XT（16GB, gfx1201）上的 RCO 量化配方与实测。
数据来自本地实验（2026-09-27），机器与模型路径已脱敏；JSON 为原始实验结果，不做模型通用榜单。

## 文件

| 文件 | 内容 |
|---|---|
| `RCO-量化配方.md` | 完整配方：为什么自己量化、tensor 类型分配、转换命令、校验、产物清单、适用模型 |
| `tensor-types-rco-851.txt` | 851 个 tensor 的类型分配表（无 MTP 头版） |
| `tensor-types-rco-866-with-mtp.txt` | 866 个 tensor 的类型分配表（含 15 个 `blk.64.*` MTP 头） |
| `2026-09-27-capacity.json` | 14 个预设的 GGUF 元数据容量计算（非 GPU 峰值实测） |
| `2026-09-27-kv-summary.json` | KV 量化 q8/q5/q4 三档 × 9K/52K 的实测汇总（含原始文件 SHA256 引用） |
| `2026-09-27-q5val-summary-v2.json` | q5/q8 @32K/52K 质量矩阵（英文/中文/工具调用命中率） |

## 核心结论

- **量化配方比硬件/框架更重要**：同一 27B 混合注意力模型，批量转的 Q3_K_L 仅 5.8 tok/s，
  按 RCO 配方（把 GDN 的 `ssm_alpha/beta` 留在 F32/BF16）重定量后 **12.37~16.95 tok/s**（~2.9x）；
- **96 个 SSM 门控参数必须保精度**：`ssm_alpha`/`ssm_beta` 压到 3bit 会破坏 Gated DeltaNet 的
  逐 token 递归状态更新（这是 RCO 配方与批量转换的唯一关键差异）；
- **KV 档位**：q8_0（推荐）、q5_0、q4_0 均可用于 KVMem 长上下文（双 needle 召回不丢）；
  非对称 `K=q8_0 V=q4_0` 比对称 q8_0 decode +13.9%（见仓库 README 的 R3 组合）。

## 复现

配方文档 `RCO-量化配方.md` 含完整命令（HF 下载 → bf16 GGUF → 按 tensor 类型表重定量）。
需要 `llama-quantize`（llama.cpp 官方构建）+ imatrix（官方 `imatrix-qwen3.8-27b.gguf`，13.6 MB）。