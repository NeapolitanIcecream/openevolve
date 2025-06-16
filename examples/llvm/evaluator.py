import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

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
    "linear-algebra/kernels/gemm/gemm.c",
    "linear-algebra/kernels/gesummv/gesummv.c",
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
        "linear-algebra/kernels/syr2k/syr2k.c",
        "stencils/adi/adi.c",
        "stencils/fdtd-2d/fdtd-2d.c",
    ]

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
        str(LLVM_SRC_DIR),
    ]
    subprocess.run(cmake_cmd, cwd=LLVM_BUILD_DIR, check=True)


def _rebuild_clang() -> float:
    """增量构建 clang，返回耗时秒"""
    start = time.time()
    subprocess.run(["ninja", "clang"], cwd=LLVM_BUILD_DIR, check=True, stdout=subprocess.DEVNULL)
    return time.time() - start


def _compile_kernel(clang_path: Path, kernel_rel: str, dataset_macro: str):
    """编译并运行单个 kernel。

    返回 (compile_time_sec, exec_time_sec, bin_size_bytes)
    """
    kernel_c = POLYBENCH_DIR / kernel_rel
    kernel_dir = kernel_c.parent
    exe_path = kernel_c.with_suffix(".exe")

    compile_cmd = [
        str(clang_path),
        "-O3",
        "-DPOLYBENCH_TIME",
        f"-D{dataset_macro}",
        "-I",
        str(POLYBENCH_DIR / "utilities"),
        "-I",
        str(kernel_dir),
        str(POLYBENCH_DIR / "utilities" / "polybench.c"),
        str(kernel_c),
        "-o",
        str(exe_path),
    ]

    # 编译
    start_compile = time.time()
    subprocess.run(compile_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    compile_time = time.time() - start_compile

    # 运行并解析时间
    run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    out = run.stdout + run.stderr
    m = re.findall(r"[0-9]+\.[0-9]+", out)
    exec_time = float(m[-1]) if m else 0.0

    bin_size = exe_path.stat().st_size

    # 清理可执行文件
    exe_path.unlink(missing_ok=True)

    return compile_time, exec_time, bin_size


def _evaluate(dataset_macro: str, kernels: List[str]) -> Dict[str, float]:
    """公共评估逻辑，收集 PolyBench kernel 的编译时间、运行时间以及二进制体积。"""
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

    return {
        "compile_total_sec": compile_total,
        "runtime_total_sec": runtime_total,
        "bin_total_bytes": bin_total,
    }

# ---------------- 公开的评估接口 ----------------

BASELINE_FILE = LLVM_BUILD_DIR / "baseline_metrics.json"


def _compute_and_cache_baseline():
    """计算官方 LoopUnrollPass 的基线性能，结果写入 BASELINE_FILE"""
    if BASELINE_FILE.exists():
        return  # 已有缓存

    print("[LLVM-Evaluator] Computing baseline metrics (first run)...")
    # 确保 PolyBench 与构建目录
    _ensure_polybench_dir()
    _configure_llvm_once()

    # 还原官方实现（若已被覆盖）
    backup_path = LOOP_UNROLL_CPP.with_suffix(".orig_bak")
    if backup_path.exists():
        shutil.copy2(backup_path, LOOP_UNROLL_CPP)
    else:
        # 第一次备份
        shutil.copy2(LOOP_UNROLL_CPP, backup_path)

    # 重新构建 clang（完整构建或增量）
    baseline_build_time = _rebuild_clang()

    # 运行 MINI 与 STANDARD 数据集
    mini_metrics = _evaluate("MINI_DATASET", STAGE1_KERNELS)
    std_metrics = _evaluate("STANDARD_DATASET", STAGE2_KERNELS)

    # 计算官方实现代码体积（字节）
    loop_unroll_size_bytes = LOOP_UNROLL_CPP.with_suffix(".orig_bak").stat().st_size if LOOP_UNROLL_CPP.with_suffix(".orig_bak").exists() else LOOP_UNROLL_CPP.stat().st_size

    baseline_data = {
        "llvm_build_time_sec": baseline_build_time,

        # PolyBench kernel 运行时间
        "mini_compile_total_sec": mini_metrics["compile_total_sec"],
        "mini_runtime_total_sec": mini_metrics["runtime_total_sec"],
        "mini_bin_total_bytes": mini_metrics["bin_total_bytes"],

        "standard_compile_total_sec": std_metrics["compile_total_sec"],
        "standard_runtime_total_sec": std_metrics["runtime_total_sec"],
        "standard_bin_total_bytes": std_metrics["bin_total_bytes"],

        # 源码大小
        "loop_unroll_code_size_bytes": loop_unroll_size_bytes,
    }

    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    import json
    BASELINE_FILE.write_text(json.dumps(baseline_data))



def _load_baseline():
    """加载基线数据，若不存在或字段缺失则重新计算"""

    required_keys = {
        "llvm_build_time_sec",
        "loop_unroll_code_size_bytes",
        "mini_compile_total_sec",
        "mini_runtime_total_sec",
        "mini_bin_total_bytes",
        "standard_compile_total_sec",
        "standard_runtime_total_sec",
        "standard_bin_total_bytes",
    }

    if not BASELINE_FILE.exists():
        _compute_and_cache_baseline()

    import json
    try:
        data = json.loads(BASELINE_FILE.read_text())
        # 若缺关键字段则强制重算
        if not required_keys.issubset(data.keys()):
            raise ValueError("baseline fields outdated")
        return data
    except Exception:
        # 若读取失败或字段缺失，重新计算
        _compute_and_cache_baseline()
        return json.loads(BASELINE_FILE.read_text())


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

    # ---------- 计算各因子 ----------
    if dataset_macro == "MINI_DATASET":
        baseline_runtime = baseline.get("mini_runtime_total_sec", baseline["mini_runtime_total_sec"])
    else:
        baseline_runtime = baseline.get("standard_runtime_total_sec", baseline["standard_runtime_total_sec"])

    runtime_speedup = baseline_runtime / (cand_metrics["runtime_total_sec"] + _EPS)

    # 编译时间加速比（PolyBench kernel 编译时间）
    if dataset_macro == "MINI_DATASET":
        baseline_compile = baseline["mini_compile_total_sec"]
        baseline_binsize = baseline["mini_bin_total_bytes"]
    else:
        baseline_compile = baseline["standard_compile_total_sec"]
        baseline_binsize = baseline["standard_bin_total_bytes"]

    compile_speedup = baseline_compile / (cand_metrics["compile_total_sec"] + _EPS)

    # 二进制体积比
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


def evaluate_stage1(program_path: str):
    try:
        return _evaluate_candidate(program_path, "MINI_DATASET", STAGE1_KERNELS)
    except Exception:
        return {"error": 0.0}


def evaluate_stage2(program_path: str):
    try:
        return _evaluate_candidate(program_path, "STANDARD_DATASET", STAGE2_KERNELS)
    except Exception:
        return {"error": 0.0}


# 保持 evaluate 与 stage2 等价，方便直接调用

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
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1",
                str(POLYBENCH_DIR),
            ],
            check=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to clone PolyBench: {e}") 