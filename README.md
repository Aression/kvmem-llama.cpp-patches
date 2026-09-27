# kvmem-llama.cpp 自有补丁集

对 [KVMem / kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)（AMD RX 9070 XT 16GB 单机）
的 4 个自有补丁。用于在 ROCm/HIP 后端上编译带以下功能的 `llama-kvmem-server`：

| 补丁 | 目标文件数 | 功能 |
|---|---|---|
| `kvmem-rocm-portability.patch` | 2（`src/adapter/*`） | ROCm 编译必需：`cudaErrorMemoryAllocation` → `hipErrorMemoryAllocation`。上游 master 至今未修，不加则 HIP 编译失败 |
| `qwen35moe-kvmem-capture.patch` | 1（`llama.cpp/src/models/qwen35moe.cpp`） | 给 `qwen35moe`（MoE）注入 KV 捕获调用（5 处），修复长文检索不生效（P-A） |
| `ngram-self-speculation.patch` | 4（`tools/*`） | 放开 `ngram-*` 免 draft 自投机（`--spec-type ngram-mod` 等），长引用回抄任务 decode +1.5x（B1） |
| `quest-minmax-scoring.patch` | 10（`kvmem/*`、`src/adapter/*`、`tools/*`、`scripts/build-hip.bat`、`llama.cpp/src/llama-graph.cpp`） | Quest 式块内 min/max 上界打分（`--kvmem-method bound`），budget 减半召回不丢（B2）；并给 `build-hip.bat` 加 `SKIP_KVMEM_PATCH=1` 开关 |

## 基线

- KVMem：`04878610a713d116cb525aaa749e2bb5fe08d59c`（VERSION 0.16.0-rc3）
- llama.cpp submodule pin：`b81c99b479d4c24e5eeca10de99032ebd343ef8f`
- 上游累积补丁已应用：`scripts/apply-patches.sh`（`patches/llama-kvmem-current.patch` 等）

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
git apply ..\kvmem-patches\patches\kvmem-rocm-portability.patch
git apply ..\kvmem-patches\patches\qwen35moe-kvmem-capture.patch
git apply ..\kvmem-patches\patches\ngram-self-speculation.patch
git apply ..\kvmem-patches\patches\quest-minmax-scoring.patch

:: 4. 构建（build-hip.bat 含 SKIP_KVMEM_PATCH 检测：
::    因 quest 补丁改了 llama.cpp/src/llama-graph.cpp，重建时须 set SKIP_KVMEM_PATCH=1）
set ROCM_PATH=D:\path\to\rocm
set SKIP_KVMEM_PATCH=1
scripts\build-hip.bat
:: 产物在 build-hip-win\bin\llama-kvmem-server.exe
```

Linux：`scripts/build-rocm.py --linux`，同样先 `apply-patches.sh` 再套本仓库补丁。

## 验证

- 二进制 `--list-devices` 识别 gfx1201 / ROCm0；
- 加载日志出现 `attn_layers=` 与 `KVMEM_*` 诊断行；
- 启动需动态链接：PATH 前置 ROCm SDK 的 `_rocm_sdk_devel\bin` 与 `_rocm_sdk_libraries\bin`，否则静默无输出（0xC0000135）。

## 实测（本机 27B dense + MoE，同批次）

| 配置 | decode | 说明 |
|---|---|---|
| `-ctk q8_0 -ctv q4_0 --kvmem-budget 4096 --spec-type draft-mtp --spec-draft-n-max 3` | **37.8 tok/s**（基线 23.2，+63%） | R3 组合（B2R 矩阵实测） |
| `--spec-type ngram-mod`（长引用回抄） | +1.5x | B1 验收 |
| `--kvmem-method bound` | budget 8192→4096 双针仍中 | B2 验收 |

> 二进制 92.5 MB 不在本仓库；由上述补丁 + 构建流程复现，产物 SHA256 见各补丁对应构建记录。