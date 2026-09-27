# kvmem-llama.cpp 自有补丁集 + 量化实测

对 [KVMem / kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)（AMD RX 9070 XT 16GB 单机）
的 4 个自有补丁，以及 RCO 量化配方与实测数据。

| 目录 | 内容 |
|---|---|
| `patches/` | 4 个自有补丁（ROCm 移植 / MoE 捕获 / ngram 投机 / Quest 打分） |
| `quantization/` | RCO 量化配方文档 + tensor 类型分配表 + KV 档位实测数据 |

## 上游版本（基线）

| 项 | 值 |
|---|---|
| 上游仓库 | [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) |
| 上游发行版 | **v0.16.0-rc3**（[release](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3)） |
| KVMem commit | `04878610a713d116cb525aaa749e2bb5fe08d59c` |
| llama.cpp submodule pin | `b81c99b479d4c24e5eeca10de99032ebd343ef8f` |
| 上游累积补丁 | `patches/llama-kvmem-current.patch`（SHA256 前缀 `2b11d16f`） |

## 补丁

| 补丁 | 目标文件数 | 功能 |
|---|---|---|
| `kvmem-rocm-portability.patch` | 2（`src/adapter/*`） | ROCm 编译必需：`cudaErrorMemoryAllocation` → `hipErrorMemoryAllocation`。上游 master 至今未修，不加则 HIP 编译失败 |
| `qwen35moe-kvmem-capture.patch` | 1（`llama.cpp/src/models/qwen35moe.cpp`） | 给 `qwen35moe`（MoE）注入 KV 捕获调用（5 处），修复长文检索不生效（P-A） |
| `ngram-self-speculation.patch` | 4（`tools/*`） | 放开 `ngram-*` 免 draft 自投机（`--spec-type ngram-mod` 等），长引用回抄任务 decode +1.5x（B1） |
| `quest-minmax-scoring.patch` | 10（`kvmem/*`、`src/adapter/*`、`tools/*`、`scripts/build-hip.bat`、`llama.cpp/src/llama-graph.cpp`） | Quest 式块内 min/max 上界打分（`--kvmem-method bound`），budget 减半召回不丢（B2）；并给 `build-hip.bat` 加 `SKIP_KVMEM_PATCH=1` 开关 |

补丁路径相对 kvmem-src 仓库根（如 `src/adapter/...` 即 `<repo>/src/adapter/...`）。

## 使用方式（Windows + ROCm）

```bat
:: 1. 克隆上游并 checkout 基线
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
git checkout 04878610a713d116cb525aaa749e2bb5fe08d59c
git submodule update --init --recursive   :: llama.cpp 到 b81c99b4

:: 2. 应用上游累积补丁（幂等；重复运行安全）
scripts\apply-patches.sh

:: 3. 应用本仓库补丁（顺序任意，互不重叠；均 git apply 干净）
git apply ..\kvmem-llama.cpp-patches\patches\kvmem-rocm-portability.patch
git apply ..\kvmem-llama.cpp-patches\patches\qwen35moe-kvmem-capture.patch
git apply ..\kvmem-llama.cpp-patches\patches\ngram-self-speculation.patch
git apply ..\kvmem-llama.cpp-patches\patches\quest-minmax-scoring.patch

:: 4. 构建（build-hip.bat 含 SKIP_KVMEM_PATCH 检测：
::    因 quest 补丁改了 llama.cpp/src/llama-graph.cpp，重建时须 set SKIP_KVMEM_PATCH=1）
set ROCM_PATH=D:\path\to\rocm
set SKIP_KVMEM_PATCH=1
scripts\build-hip.bat
:: 产物在 build-hip-win\bin\llama-kvmem-server.exe
```

Linux：`scripts/build-rocm.py --linux`，同样先 `apply-patches.sh` 再套本仓库补丁。

## Release 二进制

本仓库 [Releases](../../releases) 提供预编译的 **Windows x64 ROCm/HIP（gfx1201）** 二进制，
对应上述 4 补丁全部应用后的产物（构建工具链：ROCm 10.0.0 公开 Windows SDK + AMD clang 23 + MSVC 14.42）：

- 文件名：`llama-kvmem-server.exe`（约 92.5 MB）
- SHA256：见 Release asset 同目录的 `SHA256SUMS.txt`
- 运行要求：
  - GPU：AMD RX 9070 XT（gfx1201，16GB）；`--device ROCm0`、`HIP_VISIBLE_DEVICES=0`
  - 动态链接：PATH 前置 ROCm SDK 的 `_rocm_sdk_devel\bin` 与 `_rocm_sdk_libraries\bin`，否则静默无输出（0xC0000135）
  - 环境：`ROCBLAS_USE_HIPBLASLT=0`（RDNA4 上 hipBLASLt BF16 路径失效，Tensile 后端稳定）

## 量化实测

见 [quantization/README.md](quantization/README.md)。核心结论：

- **RCO 配方比硬件/框架更重要**：同一 27B 混合注意力模型，批量转的 Q3_K_L 仅 5.8 tok/s，
  按 RCO 配方（把 GDN 的 `ssm_alpha/beta` 留在 F32/BF16）重定量后 **12.37~16.95 tok/s**（~2.9x）；
- **96 个 SSM 门控参数必须保精度**：压到 3bit 会破坏 Gated DeltaNet 的逐 token 递归状态更新；
- **KV 档位**：q8_0（推荐）、q5_0、q4_0 均可用于 KVMem 长上下文；非对称 `K=q8_0 V=q4_0`
  比对称 q8_0 decode +13.9%。

## 实测（本机 27B dense + MoE，同批次）

| 配置 | decode | 说明 |
|---|---|---|
| `-ctk q8_0 -ctv q4_0 --kvmem-budget 4096 --spec-type draft-mtp --spec-draft-n-max 3` | **37.8 tok/s**（基线 23.2，+63%） | R3 组合（B2R 矩阵实测） |
| `--spec-type ngram-mod`（长引用回抄） | +1.5x | B1 验收 |
| `--kvmem-method bound` | budget 8192→4096 双针仍中 | B2 验收 |