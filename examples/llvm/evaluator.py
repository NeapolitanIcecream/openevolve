import os
import re
import shutil
import subprocess
import time
import platform
import difflib  # 新增：用于生成 diff
import statistics
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, cast, Any
import concurrent.futures  # 新增：并行评测支持
import multiprocessing  # 新增：自定义 mp context 以解决 macOS spawn 的 pickling 问题
import logging
from logging.handlers import QueueHandler, QueueListener

# -------------------------
# 用户可通过环境变量覆盖以下路径
# -------------------------
LLVM_SRC_DIR = Path(os.environ.get("LLVM_SRC_DIR", "/Users/neopolitan/Gits/llvm-project"))
LLVM_BUILD_DIR = Path(os.environ.get("LLVM_BUILD_DIR", LLVM_SRC_DIR / "build-evolve"))
POLYBENCH_DIR = Path(os.environ.get("POLYBENCH_DIR", os.getcwd() + "/polybench"))

# 目标文件在官方 LLVM 源码中的位置
LOOP_UNROLL_CPP = LLVM_SRC_DIR / "llvm" / "lib" / "Transforms" / "Scalar" / "LoopUnrollPass.cpp"

# 评估用到的 PolyBench kernel 列表（相对 POLYBENCH_DIR）
# Stage-1 用更少的 kernel，加速筛选；Stage-2 用全量 30 个 kernel
STAGE1_KERNELS = [
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/gesummv/gesummv.c",
    "stencils/jacobi-2d/jacobi-2d.c",
    "datamining/correlation/correlation.c",
    "medley/floyd-warshall/floyd-warshall.c",
]

# 若 kernel 列表文件存在，则动态读取全量列表
_STAGE2_LIST_FILE = POLYBENCH_DIR / "utilities" / "benchmark_list"
if _STAGE2_LIST_FILE.exists():
    with open(_STAGE2_LIST_FILE) as f:
        STAGE2_KERNELS = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
else:
    # 回退到手写列表，确保脚本可运行
    STAGE2_KERNELS = STAGE1_KERNELS + [
        "linear-algebra/kernels/2mm/2mm.c",
        "linear-algebra/kernels/3mm/3mm.c",
        "linear-algebra/blas/syr2k/syr2k.c",
        "stencils/adi/adi.c",
        "stencils/fdtd-2d/fdtd-2d.c",
    ]

# ---------- 调试辅助 ----------
_DEBUG = os.environ.get("LLVM_EVAL_DEBUG", "0") == "1"

# ---------- 集中式日志系统（QueueHandler/QueueListener） ----------
_LOG_QUEUE: Optional[multiprocessing.Queue] = None  # type: ignore[assignment]
_QUEUE_LISTENER: Optional[QueueListener] = None


def _start_logging():
    """在主进程初始化集中式日志，仅调用一次。"""
    global _LOG_QUEUE, _QUEUE_LISTENER
    if _LOG_QUEUE is not None:
        return  # 已初始化

    # 队列 + 监听器
    _LOG_QUEUE = multiprocessing.Queue(-1)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('[%(processName)s] %(message)s'))
    _QUEUE_LISTENER = QueueListener(_LOG_QUEUE, stream_handler)
    _QUEUE_LISTENER.start()

    # 主进程直接输出到控制台
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if _DEBUG else logging.INFO)
    root_logger.addHandler(stream_handler)


def _worker_logging_init(queue: "multiprocessing.Queue"):
    """子进程初始化：把日志发送到主进程队列。"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if _DEBUG else logging.INFO)
    # 移除可能存在的 Handler，避免重复输出
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.addHandler(QueueHandler(queue))

# ---------------- 性能波动抵抗 ----------------
# 通过对每个 kernel 可执行文件进行多次运行并取中位数，降低系统噪声对评测结果的影响。
# 运行次数可通过环境变量 POLYBENCH_RUNS 配置，默认为 1 次。
_EXEC_REPEAT = max(1, int(os.environ.get("POLYBENCH_RUNS", "3")))

# warm-up 次数（不计入统计）
_WARMUP_RUNS = max(0, int(os.environ.get("POLYBENCH_WARMUP", "0")))

# 是否裁剪极端值（当 _EXEC_REPEAT ≥5 时启用）
_TRIM_EXTREMES = os.environ.get("POLYBENCH_TRIM", "1") == "1"

# CPU 亲和性与 nice 设置（可选）
_CPU_PIN_ENABLED = os.environ.get("POLYBENCH_CPU_PIN", "0") == "1" and platform.system() == "Linux"
_CPU_PIN_CORE = os.environ.get("POLYBENCH_CPU_CORE", "0")
_NICE_VALUE = os.environ.get("POLYBENCH_NICE")  # 若未设置则为 None

# 是否启用性能回退早退机制（默认关闭，可通过环境变量 LLVM_EVAL_EARLY_STOP=1 开启）
_ENABLE_EARLY_STOP = os.environ.get("LLVM_EVAL_EARLY_STOP", "0") == "1"

def _log(msg: str):
    """仅在环境变量 LLVM_EVAL_DEBUG=1 时打印调试信息（集中式日志）。"""
    if _DEBUG:
        logging.getLogger().info(msg)

# ---------- 工具函数 ----------

def _strip_evolve_markers(code: str) -> str:
    """删除 # EVOLVE-BLOCK-* 标记行"""
    cleaned_lines = []
    for line in code.splitlines():
        if "EVOLVE-BLOCK-START" in line or "EVOLVE-BLOCK-END" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines) + "\n"


def _write_candidate_to_llvm(src_path: str):
    """读取候选代码，清理标记后写入官方 LoopUnrollPass.cpp"""
    candidate_code = Path(src_path).read_text()
    cleaned_code = _strip_evolve_markers(candidate_code)

    # --------- 提前比较，若内容完全一致则跳过写入，避免无意义的增量编译 ---------
    try:
        if LOOP_UNROLL_CPP.exists() and LOOP_UNROLL_CPP.read_text() == cleaned_code:
            _log("候选代码与现有 LoopUnrollPass.cpp 完全一致，跳过覆写与后续重编译")
            return  # 直接退出，无需备份/写入
    except Exception as cmp_exc:
        # 读取失败时不影响主流程，仅记录日志
        _log(f"比较现有文件内容失败: {cmp_exc}")

    # ---------- 生成 diff 统计（仅在调试模式） ----------
    if _DEBUG:
        try:
            original_code = LOOP_UNROLL_CPP.read_text()
            original_lines = original_code.splitlines()
            candidate_lines = cleaned_code.splitlines()

            diff_lines = list(
                difflib.unified_diff(
                    original_lines,
                    candidate_lines,
                    fromfile="original",
                    tofile="candidate",
                    n=3,
                )
            )

            # 统计变更行（忽略 diff 头部 @@ 行）
            changed_count = sum(
                1
                for ln in diff_lines
                if ln.startswith("+") or ln.startswith("-") and not ln.startswith("@@")
            )

            _log(
                "候选代码行数: {}；原始代码行数: {}；diff 变更行数: {}".format(
                    len(candidate_lines), len(original_lines), changed_count
                )
            )

            # 若差异过大（例如与原文件行数相当），提示可能是全文件替换
            if changed_count > 0.8 * len(original_lines):
                _log("⚠️ 检测到大规模变更，候选可能为全文件替换！")

            # 仅打印前若干行 diff 作为预览，避免日志过长
            preview_cnt = 30
            if diff_lines:
                _log(
                    "Diff 预览（前 {} 行）:\n{}".format(
                        preview_cnt, "\n".join(diff_lines[:preview_cnt])
                    )
                )
        except Exception as diff_exc:
            _log(f"生成 diff 统计失败: {diff_exc}")

    # 备份一次原始文件（首次运行时）
    backup_path = LOOP_UNROLL_CPP.with_suffix(".orig_bak")
    if not backup_path.exists():
        shutil.copy2(LOOP_UNROLL_CPP, backup_path)

    # 写入新的实现
    LOOP_UNROLL_CPP.write_text(cleaned_code)


def _configure_llvm_once():
    """若 build 目录不存在则调用 cmake"""
    if LLVM_BUILD_DIR.exists():
        return

    LLVM_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 获取本机架构（例如 X86 或 AArch64）
    try:
        host_triple = subprocess.check_output(["llvm-config", "--host-target"], text=True).strip()
        native_arch = host_triple.split("-")[0] if "-" in host_triple else host_triple
    except Exception:
        native_arch = "X86"  # 回退

    cmake_cmd = [
        "cmake",
        "-G",
        "Ninja",
        "-DLLVM_ENABLE_PROJECTS=clang",
        f"-DLLVM_TARGETS_TO_BUILD={native_arch}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_DOCS=OFF",
    ]

    # --------- 可选：启用 ccache 以加速后续重复构建 ---------
    # 通过环境变量 LLVM_USE_CCACHE=0 可以显式关闭
    if os.environ.get("LLVM_USE_CCACHE", "1") != "0":
        cmake_cmd += [
            "-DLLVM_CCACHE_BUILD=ON",  # LLVM 内置支持
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
        _log("已在 CMake 配置中启用 ccache")

    # --------- 启用 lld 作为链接器（macOS/Linux 通用） ---------
    if os.environ.get("LLVM_USE_LLD", "1") != "0":
        cmake_cmd += ["-DLLVM_ENABLE_LLD=ON"]
        _log("已在 CMake 配置中启用 lld")

    cmake_cmd.append(str(LLVM_SRC_DIR))

    _run_subprocess(cmake_cmd, cwd=LLVM_BUILD_DIR)


def _rebuild_clang() -> float:
    """增量构建 clang，返回耗时秒"""
    # 支持通过环境变量 LLVM_NINJA_JOBS 控制并行度；
    # 若未设置则让 Ninja 使用默认并行度（等于 CPU 核心数）。
    ninja_jobs = os.environ.get("LLVM_NINJA_JOBS")
    ninja_cmd = ["ninja"]
    if ninja_jobs:
        ninja_cmd += ["-j", str(ninja_jobs)]
    ninja_cmd += ["LLVMScalarOpts", "clang"]

    _log("开始增量构建 clang（{}）…".format(" ".join(ninja_cmd)))
    start = time.time()
    _run_subprocess(ninja_cmd, cwd=LLVM_BUILD_DIR)
    duration = time.time() - start
    _log(f"clang 构建完成，用时 {duration:.2f}s")
    return duration


def _compile_kernel(clang_path: Path, kernel_rel: str, dataset_macro: str):
    """编译并运行单个 kernel。

    返回 (compile_time_sec, exec_time_sec, bin_size_bytes)
    """
    _log(f"\n==== 开始处理 kernel: {kernel_rel} (数据集: {dataset_macro}) ====")
    kernel_c = POLYBENCH_DIR / kernel_rel
    kernel_dir = kernel_c.parent
    exe_path = kernel_c.with_suffix(".exe")

    # ----------------------
    # 构建编译命令
    # ----------------------
    compile_cmd: List[str] = [str(clang_path)]

    # 编译优化与数据集
    compile_cmd += [
        "-O3",
        "-std=c99",  # PolyBench 大多使用 C99 语法
        "-DPOLYBENCH_TIME",
        f"-D{dataset_macro}",
    ]

    # macOS 下，刚编译出的 clang 位于构建目录，尚未安装到系统路径。
    # 若直接调用，往往找不到系统头文件（如 stdio.h）。
    # 这里通过 xcrun 查询 SDK 路径并显式指定 sysroot，确保可以找到系统头文件。
    if platform.system() == "Darwin":
        try:
            sdk_path = subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                text=True,
            ).strip()
            if sdk_path:
                compile_cmd += ["-isysroot", sdk_path]
        except Exception:
            # 若 xcrun 不可用，则继续，期待 Clang 自行找到头文件
            pass

    # PolyBench 及 kernel 的头文件搜索路径
    compile_cmd += [
        "-I",
        str(POLYBENCH_DIR / "utilities"),
        "-I",
        str(kernel_dir),
    ]

    # 源文件以及输出
    compile_cmd += [
        str(POLYBENCH_DIR / "utilities" / "polybench.c"),
        str(kernel_c),
        "-o",
        str(exe_path),
    ]

    # 编译
    if _DEBUG:
        _log("编译命令: " + " ".join(compile_cmd))
    start_compile = time.time()
    _run_subprocess(compile_cmd)
    compile_time = time.time() - start_compile
    _log(f"编译完成，用时 {compile_time:.2f}s")

    # 运行可执行文件
    # ------------------------------
    # 拼装执行命令（亲和性 + nice）
    # ------------------------------
    base_cmd: List[str] = [str(exe_path)]
    if _CPU_PIN_ENABLED:
        base_cmd = [
            "taskset",
            "-c",
            str(_CPU_PIN_CORE),
        ] + base_cmd

    if _NICE_VALUE is not None:
        base_cmd = ["nice", "-n", str(_NICE_VALUE)] + base_cmd

    # ------------------------------
    # warm-up（不计入统计）
    # ------------------------------
    if _WARMUP_RUNS > 0:
        _log(f"进行 warm-up { _WARMUP_RUNS } 次…")
        for w in range(_WARMUP_RUNS):
            subprocess.run(base_cmd, capture_output=True, text=True, check=True)

    # ------------------------------
    # 正式多次执行并采样
    # ------------------------------
    _log(f"开始正式执行，共 {_EXEC_REPEAT} 次…")

    exec_times: List[float] = []
    for i in range(_EXEC_REPEAT):
        if _DEBUG:
            _log(f"  第 {i + 1}/{_EXEC_REPEAT} 次…")

        run_result = subprocess.run(base_cmd, capture_output=True, text=True, check=True)
        out = run_result.stdout + run_result.stderr
        m = re.findall(r"[0-9]+\.[0-9]+", out)
        t = float(m[-1]) if m else 0.0
        exec_times.append(t)

        if _DEBUG:
            _log(f"    耗时 {t:.2f}s")

    # 裁剪极端值（若启用且样本足够）
    exec_times_sorted = sorted(exec_times)
    if _TRIM_EXTREMES and len(exec_times_sorted) >= 3:
        trimmed = exec_times_sorted[1:-1]  # 去掉 min 与 max
        exec_used = trimmed
        _log(
            f"已裁剪极端值: min={exec_times_sorted[0]:.2f}s, "
            f"max={exec_times_sorted[-1]:.2f}s"
        )
    else:
        exec_used = exec_times_sorted

    # 使用平均数作为代表值（已先裁剪极端值）
    exec_time = statistics.mean(exec_used)

    _log(
        "执行完毕，中位耗时 {:.2f}s (采样: {})".format(
            exec_time, ", ".join(f"{et:.2f}" for et in exec_times)
        )
    )

    bin_size = exe_path.stat().st_size
    _log(f"二进制体积: {bin_size} 字节")

    # 清理可执行文件
    exe_path.unlink(missing_ok=True)

    return compile_time, exec_time, bin_size


def _evaluate(dataset_macro: str, kernels: List[str]) -> Dict[str, Any]:
    """公共评估逻辑，收集 PolyBench kernel 的编译时间、运行时间以及二进制体积。

    支持通过环境变量 POLYBENCH_PARALLELISM 控制并行度：
        - 未设置或设为 1 时保持串行，获得最稳定的性能数据；
        - >1 时启用并行，对不同 kernel 进行并行编译/运行，以加速整体评测。

    由于并行运行可能导致缓存/调度干扰，建议并行度不要超过物理核心数的一半，
    或者根据具体硬件与负载酌情调整。
    """
    _log(f"开始评估数据集 {dataset_macro}，共 {len(kernels)} 个 kernels…")
    clang_bin = LLVM_BUILD_DIR / "bin" / "clang"
    if not clang_bin.exists():
        raise RuntimeError("clang binary 不存在，LLVM 构建可能失败")

    compile_total = 0.0
    runtime_total = 0.0
    bin_total = 0
    # 记录每个 kernel 的执行时间，便于后续计算几何平均 speed-up
    exec_times_per_kernel: Dict[str, float] = {}

    # ------------------------------
    # 提前退出基线准备
    # ------------------------------
    prefix = "mini" if dataset_macro == "MINI_DATASET" else "standard"
    _baseline_exec_times: Dict[str, float] = _load_baseline().get(f"{prefix}_exec_times", {})  # type: ignore[assignment]

    # ------------------------------
    # 并行度配置
    # ------------------------------
    try:
        parallelism = max(1, int(os.environ.get("POLYBENCH_PARALLELISM", "4")))
    except ValueError:
        parallelism = 1
    if parallelism > 1:
        _log(f"启用并行评测，worker 数: {parallelism}")

        # 使用进程池以避免 GIL 影响，并减少不同 kernel 之间的共享状态冲突
        # macOS / Python 3.8+ 默认采用 spawn，会导致无法 pickle 动态加载的函数对象。
        # 这里显式使用 "fork"（非 Windows 平台支持）或回退到默认 spawn，
        # 通过 mp_context 参数传入自定义的 multiprocessing.Context。
        if platform.system() != "Windows":
            mp_ctx = multiprocessing.get_context("fork")
        else:
            mp_ctx = multiprocessing.get_context("spawn")

        # 确保日志系统已就绪
        _start_logging()

        # 显式管理进程池，便于早退时立即取消剩余任务而不被 "with ... as pool" 的
        # 隐式 shutdown(wait=True) 阻塞。
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=parallelism,
            mp_context=mp_ctx,
            initializer=_worker_logging_init,
            initargs=(cast(multiprocessing.Queue, _LOG_QUEUE),),  # type: ignore[arg-type]
        )

        early_stopped = False  # 标记是否触发早退，用于决定 finally 中的等待策略

        try:
            future_to_kernel = {
                pool.submit(_compile_kernel, clang_bin, k, dataset_macro): k for k in kernels
            }

            for fut in concurrent.futures.as_completed(future_to_kernel):
                k_name = future_to_kernel[fut]
                try:
                    c_time, r_time, b_size = fut.result()
                except Exception as exc:
                    # 发生异常时立即取消剩余任务，然后向上抛出，避免长时间等待。
                    early_stopped = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"Kernel {k_name} 评测失败: {exc}") from exc

                compile_total += c_time
                runtime_total += r_time
                bin_total += b_size
                exec_times_per_kernel[k_name] = r_time

                # 提前退出（可选）：若存在基线且性能回退超过 5%
                b_t = _baseline_exec_times.get(k_name)
                if _ENABLE_EARLY_STOP and b_t is not None and b_t > 0 and r_time / b_t > 1.05:
                    regress_pct = (r_time / b_t - 1) * 100
                    _log(
                        f"Kernel {k_name} 性能回退 {regress_pct:.2f}% (>5%)，提前终止评测 (软失败)"
                    )

                    # 取消所有未完成任务，立即返回
                    early_stopped = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    return {
                        "_early_stop": 1,
                        "compile_total_sec": compile_total,
                        "runtime_total_sec": runtime_total,
                        "bin_total_bytes": bin_total,
                        "exec_times": exec_times_per_kernel,
                    }
        finally:
            # 根据是否早退决定是否等待所有任务结束。若已早退，则无需阻塞等待。
            pool.shutdown(wait=not early_stopped)
    else:
        # 串行执行（默认）
        for k in kernels:
            c_time, r_time, b_size = _compile_kernel(clang_bin, k, dataset_macro)
            compile_total += c_time
            runtime_total += r_time
            bin_total += b_size
            exec_times_per_kernel[k] = r_time

            # 提前退出（可选）：若存在基线且性能回退超过 5%
            b_t = _baseline_exec_times.get(k)
            if _ENABLE_EARLY_STOP and b_t is not None and b_t > 0 and r_time / b_t > 1.05:
                regress_pct = (r_time / b_t - 1) * 100
                _log(
                    f"Kernel {k} 性能回退 {regress_pct:.2f}% (>5%)，提前终止评测 (软失败)"
                )
                return {
                    "_early_stop": 1,
                    "compile_total_sec": compile_total,
                    "runtime_total_sec": runtime_total,
                    "bin_total_bytes": bin_total,
                    "exec_times": exec_times_per_kernel,
                }

    _log(
        f"数据集 {dataset_macro} 完成：总编译时间 {compile_total:.2f}s，"
        f"总运行时间 {runtime_total:.2f}s，累积二进制体积 {bin_total} 字节"
    )

    return {
        "compile_total_sec": compile_total,
        "runtime_total_sec": runtime_total,
        "bin_total_bytes": bin_total,
        "exec_times": exec_times_per_kernel,
    }

# ----------------------------
# 进程内共享基线缓存
# ----------------------------
# 在 OpenEvolve 的评估过程中，evaluation_module 会被反复以新的 module 对象动态
# 载入。若直接在本文件使用普通的模块级全局变量来保存基线，则每次加载都会重置，
# 导致无法利用之前已测得的 baseline。
#
# 这里通过在 `sys.modules` 注册一个自定义的临时模块来保存共享状态。由于 Python
# 会对同名 module 进行缓存，不同次加载本 evaluator 文件时，只要进程未结束，
# 都能拿到同一份 `shared_state`，从而实现基线跨加载持久化，而无需写入磁盘。

_SHARED_MOD = "_llvm_evaluator_shared_state"

if _SHARED_MOD not in sys.modules:
    shared_state = types.ModuleType(_SHARED_MOD)
    # 使用 type: ignore 避免静态检查器关于动态属性的告警
    shared_state.BASELINE_CACHE = {}  # type: ignore[attr-defined]
    sys.modules[_SHARED_MOD] = shared_state
else:
    shared_state = sys.modules[_SHARED_MOD]

# `shared_state.BASELINE_CACHE` 即为跨 module 实例共享的字典。
_BASELINE_CACHE: Dict[str, Any] = shared_state.BASELINE_CACHE

# 返回当前进程基线缓存（若为空则说明尚未初始化）
def _load_baseline() -> Dict[str, Any]:
    return _BASELINE_CACHE

# ---------------- 重写后的评估接口 ----------------

# -------------------------
# 评分相关全局配置
# -------------------------
# 权重含义：
#   runtime_speedup   —— 运行时间加速比（baseline_runtime / candidate_runtime）
#   compile_speedup   —— PolyBench kernel 编译时间加速比（baseline_compile / candidate_compile）
#   code_size_ratio   —— 二进制体积比（baseline_binsize / candidate_binsize）
# 线性组合的形式为 Σ w_i * factor_i，默认仅考虑运行时间。
SCORING_WEIGHTS = {
    "runtime_speedup": 0.9,
    "compile_speedup": 0.0,
    "code_size_ratio": 0.1,
}

_EPS = 1e-9  # 防止除零

def _evaluate_candidate(
    program_path: str,
    dataset_macro: str,
    kernels: List[str],
    rebuild_llvm: bool = True,
):
    """公共流程：写入候选 → 评估 → 计算归一化指标

    评分因子：
        - runtime_speedup   = baseline_runtime / candidate_runtime
        - compile_speedup   = baseline_compile / candidate_compile
        - code_size_ratio   = baseline_binsize / candidate_binsize

    最终 fitness = Σ weight_i * factor_i
    """

    _ensure_polybench_dir()
    baseline = _load_baseline()

    # 启动集中日志系统（主进程一次性）
    _start_logging()

    # 写入候选代码
    _write_candidate_to_llvm(program_path)

    # ---------- 编译 LLVM（可选）
    _configure_llvm_once()

    # 如果禁用重建，则仅当 clang 不存在时才强制构建
    need_rebuild = rebuild_llvm or not (LLVM_BUILD_DIR / "bin" / "clang").exists()

    if need_rebuild:
        build_time = _rebuild_clang()
    else:
        build_time = 0.0  # 跳过重建

    # ---------- PolyBench 测试
    cand_metrics = _evaluate(dataset_macro, kernels)

    # 若 _evaluate 提前标记软失败，则直接返回 0 分，不更新基线
    if cand_metrics.get("_early_stop"):
        return {
            "runtime_speedup": 0.0,
            "compile_speedup": 0.0,
            "code_size_ratio": 0.0,
            "combined_score": 0.0,
        }

    # ---------- 计算各因子与动态更新基线 ----------
    prefix = "mini" if dataset_macro == "MINI_DATASET" else "standard"

    runtime_key = f"{prefix}_runtime_total_sec"
    compile_key = f"{prefix}_compile_total_sec"
    bin_key = f"{prefix}_bin_total_bytes"
    exec_times_key = f"{prefix}_exec_times"

    # 若当前基线缺失相应字段，则将候选结果写入内存基线
    updated_baseline = False
    if runtime_key not in baseline:
        baseline[runtime_key] = cand_metrics["runtime_total_sec"]
        updated_baseline = True
        _log(f"基线缺失 {runtime_key}，使用候选值: {cand_metrics['runtime_total_sec']:.2f}s")
    if compile_key not in baseline:
        baseline[compile_key] = cand_metrics["compile_total_sec"]
        updated_baseline = True
        _log(f"基线缺失 {compile_key}，使用候选值: {cand_metrics['compile_total_sec']:.2f}s")
    if bin_key not in baseline:
        baseline[bin_key] = cand_metrics["bin_total_bytes"]
        updated_baseline = True
        _log(f"基线缺失 {bin_key}，使用候选值: {cand_metrics['bin_total_bytes']} bytes")
    if exec_times_key not in baseline:
        baseline[exec_times_key] = cand_metrics["exec_times"]
        updated_baseline = True
        _log(f"基线缺失 {exec_times_key}，已记录候选的 per-kernel 执行时间")

    # 其他可能缺失的通用字段
    if "llvm_build_time_sec" not in baseline:
        baseline["llvm_build_time_sec"] = build_time
        updated_baseline = True

    if "loop_unroll_code_size_bytes" not in baseline:
        try:
            baseline["loop_unroll_code_size_bytes"] = Path(program_path).stat().st_size
            updated_baseline = True
        except Exception:
            pass

    if updated_baseline:
        # 更新全局缓存
        global _BASELINE_CACHE
        _BASELINE_CACHE = baseline

    # 使用（可能已更新的）基线值计算评分因子
    baseline_runtime = baseline.get(runtime_key, cand_metrics["runtime_total_sec"])
    baseline_compile = baseline.get(compile_key, cand_metrics["compile_total_sec"])
    baseline_binsize = baseline.get(bin_key, cand_metrics["bin_total_bytes"])

    # -------------- 按 kernel 计算 runtime speed-up 的几何平均值 --------------
    cand_exec_times: Dict[str, float] = cand_metrics.get("exec_times", {})
    baseline_exec_times: Dict[str, float] = baseline.get(exec_times_key, {})  # type: ignore[assignment]

    per_kernel_speedups: List[float] = []
    for k in kernels:
        b_t = baseline_exec_times.get(k)
        c_t = cand_exec_times.get(k)
        if b_t is None or c_t is None or c_t <= 0:
            continue
        per_kernel_speedups.append(b_t / (c_t + _EPS))

    if per_kernel_speedups:
        runtime_speedup = statistics.geometric_mean(per_kernel_speedups)
    else:
        runtime_speedup = 1.0  # 回退

    compile_speedup = baseline_compile / (cand_metrics["compile_total_sec"] + _EPS)
    code_size_ratio = baseline_binsize / (cand_metrics["bin_total_bytes"] + _EPS)

    # ---------- 线性组合 ----------

    return {
        "runtime_speedup": runtime_speedup,
        "compile_speedup": compile_speedup,
        "code_size_ratio": code_size_ratio,
        "combined_score": (
            SCORING_WEIGHTS["runtime_speedup"] * runtime_speedup
            + SCORING_WEIGHTS["compile_speedup"] * compile_speedup
            + SCORING_WEIGHTS["code_size_ratio"] * code_size_ratio
        ),
    }


def _exception_details(e: Exception):
    import traceback
    tb = traceback.format_exc()
    return {
        "error": str(e),
        "traceback": tb,
    }


def _restore_original_llvm_file():
    """将 LoopUnrollPass.cpp 恢复为最初的备份版本（若存在）。

    该函数在评估流程结束后调用，确保不会因为候选代码覆盖而导致后续构建/使用出现异常。"""
    backup_path = LOOP_UNROLL_CPP.with_suffix(".orig_bak")
    if backup_path.exists():
        try:
            shutil.copy2(backup_path, LOOP_UNROLL_CPP)
            _log("已恢复原始 LoopUnrollPass.cpp 文件")
        except Exception as e:
            # 出现异常时仅记录，而不终止主流程
            _log(f"恢复原始 LoopUnrollPass.cpp 失败：{e}")
    else:
        # 当尚未创建备份时，说明尚未写入过候选代码，忽略即可
        pass


def evaluate_stage1(program_path: str):
    try:
        return _evaluate_candidate(program_path, "MINI_DATASET", STAGE1_KERNELS)
    except Exception as e:
        return _exception_details(e)
    finally:
        # 无论成功或失败，都尝试恢复原始文件
        _restore_original_llvm_file()


def evaluate_stage2(program_path: str):
    try:
        # Stage-2 默认复用 Stage-1 已经构建好的 LLVM，无需再次重建
        return _evaluate_candidate(
            program_path,
            "STANDARD_DATASET",
            STAGE2_KERNELS
        )
    except Exception as e:
        return _exception_details(e)
    finally:
        _restore_original_llvm_file()


# 保持 evaluate 与 stage1 等价，方便直接调用

def evaluate(program_path: str):
    return evaluate_stage2(program_path)

# ---------------- PolyBench 准备 ----------------

def _ensure_polybench_dir():
    """若 PolyBench 源码不存在则尝试克隆"""
    if POLYBENCH_DIR.exists():
        return

    print(f"PolyBench not found at {POLYBENCH_DIR}, cloning...")
    POLYBENCH_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_subprocess(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1",
                str(POLYBENCH_DIR),
            ]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to clone PolyBench: {e}")

# ---------- 子进程执行辅助 ----------

def _run_subprocess(cmd: List[str], cwd: Optional[Path] = None, silent: bool = True):
    """执行子进程命令并在失败时提供详尽的错误日志。

    参数：
        cmd    —— 要执行的命令列表
        cwd    —— 工作目录，可选
        silent —— 是否隐藏 stdout/stderr；若环境变量 LLVM_EVAL_DEBUG=1 则自动关闭 silent
    """

    _log("运行子进程命令: " + " ".join(cmd))

    # 若显式开启调试，则不隐藏输出
    if os.environ.get("LLVM_EVAL_DEBUG", "0") == "1":
        silent = False

    try:
        if silent:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            result = subprocess.run(cmd, cwd=cwd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        # 拼接更友好的错误信息
        err_lines = [
            f"[LLVM-Evaluator] Command failed: {' '.join(cmd)}",
            f"Return code: {e.returncode}",
        ]
        if hasattr(e, "stdout") and e.stdout:
            err_lines.append("stdout:\n" + e.stdout)
        if hasattr(e, "stderr") and e.stderr:
            err_lines.append("stderr:\n" + e.stderr)
        raise RuntimeError("\n".join(err_lines)) from e 