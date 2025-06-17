# LLVM LoopUnroll Evolution

本示例展示如何使用 **OpenEvolve** 对 LLVM 的 `LoopUnrollPass.cpp` 中标记为 `# EVOLVE-BLOCK-START/END` 的代码片段进行自动演化，以在 PolyBench-C 基准上提升生成程序的运行速度。

---
## 目录结构
```text
examples/llvm/
 ├─ LoopUnrollPass.cpp   # 目标文件（已插入 EVOLVE 标记）
 ├─ evaluator.py         # 评估脚本（单阶段 MINI_DATASET）
 ├─ config.yaml          # OpenEvolve 配置
 └─ README.md            # 当前文档
```

## 快速开始
```bash
cd examples/llvm
python ../../openevolve-run.py LoopUnrollPass.cpp evaluator.py --config config.yaml
```
运行时将：
1. 把候选代码写回到 LLVM 源码中的 `LoopUnrollPass.cpp`（移除 EVOLVE 标记）。
2. 以 **Ninja** 增量构建 `clang`。
3. 使用 PolyBench-C *MINI_DATASET* 评估 5 个代表性 kernel。
4. 根据运行时间加速比计算 `fitness`（参见 `evaluator.py/SCORING_WEIGHTS`）。

> ⚠️ 依赖增量编译 LLVM，单次评估仍需数分钟；请在具备足够 CPU 与磁盘性能的机器上运行。

## 依赖与路径
| 环境变量 | 含义 | 默认值 |
|----------|------|--------|
| LLVM_SRC_DIR | LLVM 源码根目录 | `/Users/neopolitan/Gits/llvm-project` |
| LLVM_BUILD_DIR | LLVM 构建输出目录 | `$LLVM_SRC_DIR/build-evolve` |
| POLYBENCH_DIR | PolyBench 根目录（若不存在会自动克隆） | `$(pwd)/polybench` |

其它依赖：`cmake`、`ninja`、`git`、以及可执行的 `llvm-config`。

## 自定义
- **评估规模**：调用 `evaluator.evaluate_stage2()` 或在 `config.yaml` 把 `cascade_evaluation` 设为 `true` 可启用 *STANDARD_DATASET* 全量评测。
- **Kernel 列表**：修改 `evaluator.py` 中 `STAGE1_KERNELS` / `STAGE2_KERNELS`。
- **评分因子**：调整 `SCORING_WEIGHTS` 改变运行时间 / 编译时间 / 二进制体积的权重。
- **并行评估**：在 `config.yaml` 中提高 `parallel_evaluations`（需自行保证硬件资源）。

祝你玩的开心，期待发掘出让编译器更快的魔法！
