# LLVM LoopUnroll Evolution Example

本示例展示如何使用 **OpenEvolve** 对 LLVM 自身的 `LoopUnrollPass.cpp` 中标记为 `# EVOLVE-BLOCK-START/END` 的代码片段进行自动演化，以提升新编译器在 PolyBench-C 基准上的运行性能。

> ⚠️ 由于需要反复 **增量编译 LLVM/Clang** 并编译执行 PolyBench 内的 30 个 kernel，单次评估耗时较长（数分钟到十几分钟，取决于 CPU 核心数与磁盘性能）。请务必在 **具备充足算力与磁盘空间** 的机器上运行。

---
## 目录结构

```text
examples/llvm/
 ├─ initial_program.cpp       # 原始 LoopUnrollPass.cpp（由脚本自动拷贝）
 ├─ evaluator.py              # 两阶段 PolyBench 评估器（MINI→STANDARD）
 ├─ config.yaml               # 进化配置文件
 └─ README.md                 # 当前文档
```

## 准备工作

1. **LLVM 源码**
   - 默认路径：`/Users/neopolitan/Gits/llvm-project`（可通过环境变量 `LLVM_SRC_DIR` 覆盖）。
   - 示例脚本将在 `LLVM_SRC_DIR/build-evolve` 目录下使用 **CMake+Ninja** 增量构建 `clang`。
2. **PolyBench-C 4.2.1**
   - 未检测到时，脚本会自动克隆到 `$(pwd)/polybench`。
3. **依赖**
   - Mac 或 Linux，需要预先安装：`cmake`、`ninja`、`git`、`llvm-config`（来自现有 LLVM 安装）。

## 运行示例

```bash
# 进入例子目录
cd examples/llvm

# 运行 OpenEvolve（假设已激活所需 python 环境）
python ../../openevolve-run.py LoopUnrollPass.cpp evaluator.py --config config.yaml
```

### 可选环境变量
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLVM_SRC_DIR` | LLVM 源码根目录 | `/Users/neopolitan/Gits/llvm-project` |
| `LLVM_BUILD_DIR` | LLVM 构建输出目录 | `$LLVM_SRC_DIR/build-evolve` |
| `POLYBENCH_DIR` | PolyBench 根目录 | `$(pwd)/polybench` |

---
## 评估流程概览

1. **代码写回**：将当前候选文件写入 `llvm/lib/Transforms/Scalar/LoopUnrollPass.cpp`，并移除进化标记行；首次运行会备份原始文件到 `*.orig_bak`。
2. **增量构建**：调用 Ninja 仅重新编译受影响的对象并重链 `clang`。
3. **两阶段 PolyBench**
   1. **Stage-1（MINI_DATASET）**：快速测试 5 个代表性 kernel，用于粗筛。
   2. **Stage-2（STANDARD_DATASET）**：对全部 kernel 精测。
4. **评分**：基于官方实现（baseline）的归一化指标，默认仅使用运行时间加速比  
   ```text
   fitness = Σ w_i × factor_i     (w_i 来源于 SCORING_WEIGHTS)
   ```
   其中 `runtime_speedup = baseline_runtime / candidate_runtime`，其它因子（编译时间、二进制体积）同理。

在 `config.yaml` 中启用了 `cascade_evaluation`，使 Stage-1 阈值达标的候选才进入 Stage-2，从而节约总时长。

---
## 常见问题

1. **编译失败 / 头文件找不到**：确认 `llvm-config` 可用，且 `LLVM_SRC_DIR` 正确。
2. **构建太慢**：
   - 开启 `ccache`/`sccache` 可显著加速。
   - 只构建本机后端（脚本已自动设置）。
3. **PolyBench 报错或执行时间为 0**：请检查 `POLYBENCH_DIR` 是否完整；可手动运行某个 kernel 验证。

---
## 自定义与扩展

- **Kernel 选择**：编辑 `evaluator.py` 中 `STAGE1_KERNELS`、`STAGE2_KERNELS` 列表即可。
- **数据集规模**：将 `MINI_DATASET`、`STANDARD_DATASET` 替换为 `SMALL_DATASET` 等宏。
- **Fitness 公式**：可在 `evaluator.py` 中修改 `SCORING_WEIGHTS` 调整各因子权重。
- **并行评估**：配置 `parallel_evaluations > 1` 以并行，但需保证硬件资源充足且 LLVM 构建目录相互隔离（默认共用 BuildDir，不建议并行构建）。

祝你玩得开心，发掘出让编译器更快的魔法改动！ 