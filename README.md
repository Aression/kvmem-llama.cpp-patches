# kvmem-llama.cpp ROCm/HIP 自有补丁 + 量化实测

对 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)（AMD RX 9070 XT 16GB, gfx1201）的 4 个自有补丁，
以及 RCO 量化配方与 KV 档位实测数据。**基线上游 v0.16.0-rc3**（kvmem@`0487861`，llama.cpp@`b81c99b4`）。

## 补丁

| 补丁 | 改动 | 用途 |
|---|---|---|
| `kvmem-rocm-portability.patch` | `src/adapter/` ×2 | HIP 编译必需：`cudaErrorMemoryAllocation` → `hipErrorMemoryAllocation`（上游未修） |
| `qwen35moe-kvmem-capture.patch` | `llama.cpp/src/models/qwen35moe.cpp` | 给 MoE 注入 KV 捕获，修复长文检索不生效（P-A） |
| `ngram-self-speculation.patch` | `tools/` ×4 | 放开 `ngram-*` 免 draft 自投机（B1） |
| `quest-minmax-scoring.patch` | `kvmem/`、`src/adapter/`、`tools/`、`scripts/build-hip.bat`、`llama.cpp/src/llama-graph.cpp` | Quest 式 min/max 上界打分（B2）+ `SKIP_KVMEM_PATCH` 构建开关 |

## 使用方式

**前置**：上游累积补丁已应用（`scripts/apply-patches.sh`，`llama-kvmem-current.patch` SHA `2b11d16f`）。

```bat
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
git checkout 04878610a713d116cb525aaa749e2bb5fe08d59c
git submodule update --init --recursive
scripts\apply-patches.sh
git apply ..\kvmem-llama.cpp-patches\patches\kvmem-rocm-portability.patch
git apply ..\kvmem-llama.cpp-patches\patches\qwen35moe-kvmem-capture.patch
git apply ..\kvmem-llama.cpp-patches\patches\ngram-self-speculation.patch
git apply ..\kvmem-llama.cpp-patches\patches\quest-minmax-scoring.patch
set SKIP_KVMEM_PATCH=1
scripts\build-hip.bat
```

**SKIP_KVMEM_PATCH=1 必需**：quest 补丁改了 `llama.cpp/src/llama-graph.cpp`，会使 build-hip.bat 的
reverse-apply 检测失败。产物在 `build-hip-win\bin\llama-kvmem-server.exe`。
Linux：`scripts/build-rocm.py --linux`，同样先 apply-patches.sh 再套补丁。

## Release 二进制

[Releases](../../releases) 提供预编译 Windows x64 ROCm/HIP（gfx1201）二进制 `llama-kvmem-server.exe`（92.5 MB）。
工具链：ROCm 10.0.0 公开 Windows SDK + AMD clang 23 + MSVC 14.42。校验：`sha256sum -c SHA256SUMS.txt`。

运行要求：`--device ROCm0`、`HIP_VISIBLE_DEVICES=0`、`ROCBLAS_USE_HIPBLASLT=0`；
PATH 必须前置 ROCm SDK `_rocm_sdk_devel\bin` 与 `_rocm_sdk_libraries\bin`（否则静默无输出 0xC0000135）。

## 量化实测

见 [quantization/](quantization/README.md)。

- **RCO 配方比硬件重要**：同一 27B，批量转 Q3_K_L 仅 5.8 tok/s；RCO 配方（SSM 门控留 F32/BF16）12.4~17.0 tok/s（**~2.9x**）；
- **KV 档位**：q8_0/q5_0/q4_0 长上下文召回不丢；非对称 `K=q8_0 V=q4_0` decode **+13.9%**；
- **R3 组合**：`-ctk q8_0 -ctv q4_0 --kvmem-budget 4096 --spec-type draft-mtp --spec-draft-n-max 3` → decode **37.8 tok/s**（基线 23.2，**+63%**）。