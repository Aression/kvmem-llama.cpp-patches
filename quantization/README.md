# RCO 量化配方与实测

Qwen3.8 系列 27B dense 在 AMD RX 9070 XT（16GB, gfx1201）上的 RCO 量化配方与 KV 档位实测。
数据来自本地实验（2026-09-27）；JSON 为原始结果，路径已脱敏，不作通用榜单。

## 文件

| 文件 | 内容 |
|---|---|
| `RCO-量化配方.md` | 配方全文：tensor 分配、转换命令、校验、产物 |
| `tensor-types-rco-851.txt` | 851 tensor 类型表（无 MTP 头） |
| `tensor-types-rco-866-with-mtp.txt` | 866 tensor 类型表（含 15 个 `blk.64.*` MTP 头） |
| `2026-09-27-capacity.json` | 14 预设 GGUF 元数据容量计算 |
| `2026-09-27-kv-summary.json` | KV q8/q5/q4 × 9K/52K 实测汇总（含源文件 SHA256） |
| `2026-09-27-q5val-summary-v2.json` | q5/q8 @32K/52K 质量矩阵（中英/工具调用命中率） |

## 核心结论

- **RCO 配方比硬件/框架更重要**：同一 27B 混合注意力模型，批量转 Q3_K_L 仅 **5.8 tok/s**；
  按 RCO 配方（GDN 的 `ssm_alpha/beta` 留 F32/BF16）重定量后 **12.37~16.95 tok/s**（~2.9x）；
- **96 个 SSM 门控参数必须保精度**：压到 3bit 破坏 Gated DeltaNet 的逐 token 递归状态更新——
  这是 RCO 配方与批量转换的唯一关键差异；
- **KV 档位**：q8_0（推荐）/ q5_0 / q4_0 均可用于 KVMem 长上下文（双 needle 召回不丢）；
  非对称 `K=q8_0 V=q4_0` 比对称 q8_0 decode **+13.9%**。

## 复现

`RCO-量化配方.md` 含完整命令：HF 下载 → bf16 GGUF（`convert_hf_to_gguf.py --no-mtp` 保留 MTP 头）
→ `llama-quantize --tensor-type-file <上表> --imatrix imatrix-qwen3.8-27b.gguf`（官方构建，CPU 侧）。