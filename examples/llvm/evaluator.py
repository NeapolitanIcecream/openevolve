import os
import re
import shutil
import subprocess
import time
import platform
import difflib  # 新增：用于生成 diff
from pathlib import Path
from typing import Dict, List, Optional

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


def _log(msg: str):
    """仅在环境变量 LLVM_EVAL_DEBUG=1 时打印调试信息"""
    if _DEBUG:
        print(f"[LLVM-Evaluator] {msg}", flush=True)

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
        str(LLVM_SRC_DIR),
    ]
    _run_subprocess(cmake_cmd, cwd=LLVM_BUILD_DIR)


def _rebuild_clang() -> float:
    """增量构建 clang，返回耗时秒"""
    _log("开始增量构建 clang（ninja clang）…")
    start = time.time()
    _run_subprocess(["ninja", "clang"], cwd=LLVM_BUILD_DIR)
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
    _log("开始执行生成的可执行文件…")
    run_result = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    out = run_result.stdout + run_result.stderr
    m = re.findall(r"[0-9]+\.[0-9]+", out)
    exec_time = float(m[-1]) if m else 0.0
    _log(f"执行完毕，用时 {exec_time:.2f}s")

    bin_size = exe_path.stat().st_size
    _log(f"二进制体积: {bin_size} 字节")

    # 清理可执行文件
    exe_path.unlink(missing_ok=True)

    return compile_time, exec_time, bin_size


def _evaluate(dataset_macro: str, kernels: List[str]) -> Dict[str, float]:
    """公共评估逻辑，收集 PolyBench kernel 的编译时间、运行时间以及二进制体积。"""
    _log(f"开始评估数据集 {dataset_macro}，共 {len(kernels)} 个 kernels…")
    clang_bin = LLVM_BUILD_DIR / "bin" / "clang"
    if not clang_bin.exists():
        raise RuntimeError("clang binary 不存在，LLVM 构建可能失败")

    compile_total = 0.0
    runtime_total = 0.0
    bin_total = 0

    for k in kernels:
        c_time, r_time, b_size = _compile_kernel(clang_bin, k, dataset_macro)
        compile_total += c_time
        runtime_total += r_time
        bin_total += b_size
    _log(
        f"数据集 {dataset_macro} 完成：总编译时间 {compile_total:.2f}s，"
        f"总运行时间 {runtime_total:.2f}s，累积二进制体积 {bin_total} 字节"
    )

    return {
        "compile_total_sec": compile_total,
        "runtime_total_sec": runtime_total,
        "bin_total_bytes": bin_total,
    }

# ---------------- 公开的评估接口 ----------------

# 仅在内存缓存基线
_BASELINE_CACHE: Dict[str, float] = {}

# 返回当前进程基线缓存（若为空则说明尚未初始化）
def _load_baseline() -> Dict[str, float]:
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
    "runtime_speedup": 1.0,
    "compile_speedup": 0.0,
    "code_size_ratio": 0.0,
}

_EPS = 1e-9  # 防止除零

def _evaluate_candidate(program_path: str, dataset_macro: str, kernels: List[str]):
    """公共流程：写入候选 → 评估 → 计算归一化指标

    评分因子：
        - runtime_speedup   = baseline_runtime / candidate_runtime
        - compile_speedup   = baseline_compile / candidate_compile
        - code_size_ratio   = baseline_binsize / candidate_binsize

    最终 fitness = Σ weight_i * factor_i
    """

    _ensure_polybench_dir()
    baseline = _load_baseline()

    # 写入候选代码
    _write_candidate_to_llvm(program_path)

    # ---------- 编译 LLVM
    _configure_llvm_once()
    build_time = _rebuild_clang()

    # ---------- PolyBench 测试
    cand_metrics = _evaluate(dataset_macro, kernels)

    # ---------- 计算各因子与动态更新基线 ----------
    prefix = "mini" if dataset_macro == "MINI_DATASET" else "standard"

    runtime_key = f"{prefix}_runtime_total_sec"
    compile_key = f"{prefix}_compile_total_sec"
    bin_key = f"{prefix}_bin_total_bytes"

    # 若当前基线缺失相应字段，则将候选结果写入内存基线
    updated_baseline = False
    if runtime_key not in baseline:
        baseline[runtime_key] = cand_metrics["runtime_total_sec"]
        updated_baseline = True
    if compile_key not in baseline:
        baseline[compile_key] = cand_metrics["compile_total_sec"]
        updated_baseline = True
    if bin_key not in baseline:
        baseline[bin_key] = cand_metrics["bin_total_bytes"]
        updated_baseline = True

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

    runtime_speedup = baseline_runtime / (cand_metrics["runtime_total_sec"] + _EPS)
    compile_speedup = baseline_compile / (cand_metrics["compile_total_sec"] + _EPS)
    code_size_ratio = baseline_binsize / (cand_metrics["bin_total_bytes"] + _EPS)

    # ---------- 线性组合 ----------
    fitness = (
        SCORING_WEIGHTS["runtime_speedup"] * runtime_speedup
        + SCORING_WEIGHTS["compile_speedup"] * compile_speedup
        + SCORING_WEIGHTS["code_size_ratio"] * code_size_ratio
    )

    return {
        "fitness": fitness,
        "runtime_speedup": runtime_speedup,
        "compile_speedup": compile_speedup,
        "code_size_ratio": code_size_ratio,
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
        return _evaluate_candidate(program_path, "STANDARD_DATASET", STAGE2_KERNELS)
    except Exception as e:
        return _exception_details(e)
    finally:
        _restore_original_llvm_file()


# 保持 evaluate 与 stage2 等价，方便直接调用

def evaluate(program_path: str):
    return evaluate_stage1(program_path)

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