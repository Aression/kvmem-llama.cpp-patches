# 04 RCO 量化配方（把任意 Qwen3.8 系模型做到 12~17 tok/s）

## 为什么要自己做

实测同一 server / 后端 / 上下文，**仅换量化**：

| 模型 | SSM 门精度 | tg |
|---|---|---|
| 批量转换的 Q3_K_L | Q3_K | 5.80 |
| RCO 逐张量配方 | BF16/F32 | 12.37 ~ **16.95** |

原因是批量转换把 **Gated DeltaNet 的门参数 `ssm_alpha` / `ssm_beta` 压到了 3bit**，
既伤质量、又让每 token 的 GDN 更新走"边解码边反量化"的慢路径。

HF 上**没有任何写作特化模型的 RCO 现成量化**（已检索 GSQ-RCO writer / RCO Hemmingway /
GSQ Novelist 等，均无结果），所以要自己量化。

## 配方来源

ISTA-DASLab 官方逐张量分配表：
`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` 的
`tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.rco-allocation.txt`

转换后用在本项目（`llama-inf\quant\`）：
- `tensor-types-rco-866-with-mtp.txt` —— 866 条，含 15 条 `blk.64.*`（MTP 头）
- `tensor-types-rco-851.txt` —— 去掉 MTP 头的版本

格式转换：`name: TYPE` -> `name=TYPE`（`llama-quantize --tensor-type-file` 要求）。

**张量名校验**：RCO 配方 866 条 vs Hemmingway-1 的 851 个张量，唯一差异就是 15 个 MTP 头，
反向缺失 0 —— 可直接套用。

类型分布：`F32=360, IQ3_S=144, IQ4_XS=96, BF16=96, IQ3_XXS=78, Q4_K=39, IQ2_S=17, Q2_K=13,
IQ2_XS=9, Q6_K=8, IQ2_XXS=5, IQ1_M=1`。**96 条 SSM 门张量全部 BF16**
（llama-quantize 会把 `ssm_alpha` 再自动升到 F32）。

## 完整流程

```bash
export HF_ENDPOINT=https://hf-mirror.com
HFD=/d/hermes_workplace/llama-inf/tools/hfd.sh

# 1) 源权重：优先 bf16/f16（质量最好）；也可用 Q8_0 GGUF 直接重定量
bash $HFD "<repo>" --tool aria2c -x 10 -j 4 --local-dir "<dir>"

# 2) HF -> GGUF bf16（不带 --no-mtp 时默认把 MTP 头打进主 GGUF）
/d/hermes_python/Scripts/python.exe /d/hermes_workplace/llama-cpp.new/convert_hf_to_gguf.py \
  "<hf_dir>" --outfile "<out_bf16.gguf>" --outtype bf16

# 3) 按 RCO 配方量化（llama-quantize 在官方 HIP 包里，纯 CPU）
cd /d/LLMs/hip-test
PATH="$(pwd):/d/mingw64/bin:$PATH" ./llama-quantize.exe \
  --imatrix "D:/LLMs/ISTA-DASLab/imatrix-qwen3.8-27b.gguf" \
  --tensor-type-file "D:/hermes_workplace/llama-inf/quant/tensor-types-rco-866-with-mtp.txt" \
  --max-buffer-size 1024 \
  "<in.gguf>" "<out.gguf>" IQ3_S 8
```

从**已量化的 Q8_0** 源重定量时加 `--allow-requantize`（Q8_0 近乎无损，代价很小）。

## 实测结果

| 源 | 产物 | 大小 | BPW | 耗时 |
|---|---|---|---|---|
| `Altworld/Hemmingway-1` bf16（54.66 GB） | `Hemmingway-1-RCO-IQ3_S-mtp.gguf` | 12.18 GB | 3.56 | 转换 ~7 min + 量化 13 min |
| `markldn/...Abliterated-Extreme-Q8_0-MTP`（29.05 GB） | `Hemmingway-1-Abliterated-RCO-IQ3_S-mtp.gguf` | 12.18 GB | 3.56 | 量化 13.4 min |

## 验证方法

1. **张量数**：866 = 851 + 15 个 MTP（`blk.64.*`）。
2. **逐张量类型比对**官方 RCO：本文产物 866 vs 866，62 处差异**全是精度更高方向**
   （`ssm_alpha` F32 vs BF16、`attn_output` Q4_K vs IQ3_S）。
3. **实测速度**：@9K 应到 12~17 tok/s（批量 K 系量化只有 5.8）。
4. `launcher-kvmem.py --dry-run` 跑显存守卫，确认权重+KV+余量放得下。

## 重要副产品：官方 HF 源里有 MTP，转换器会丢

`Altworld/Hemmingway-1` 的 HF 仓库带 `model-mtp.safetensors`（15 张量），
`config.json` 声明 `mtp_num_hidden_layers=1`，但 **mradermacher 的 GGUF 只有 851 张量、0 个 MTP**。
`convert_hf_to_gguf.py` 不带 `--no-mtp` 时默认把 MTP 头打进主 GGUF，于是重转后就有 MTP 了。

相关开关：`--mtp`（只导出 MTP 头为独立文件，做 draft）、`--no-mtp`（排除），
仅支持 Qwen3.5/3.6 与 Step3.5 文本变体。

## 想给别的模型也提速

任何 qwen35 系（Qwen3.5/3.6/3.8-27B 及其微调）都可以套这张表，只要：
1. 张量名一致（Qwen3.8-27B 系全部一致，已校验）；
2. 有对应的 imatrix（官方 `imatrix-qwen3.8-27b.gguf`，13.6 MB，可跨微调复用）；
3. MTP 头的有无决定用 866 条还是 851 条的表。
