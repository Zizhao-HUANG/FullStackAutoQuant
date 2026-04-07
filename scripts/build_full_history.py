#!/usr/bin/env python3
"""使用原始 Docker/Dolt 流水线下载全量 Qlib 数据并提取 norm_params.pkl。

流程：
  1. 运行 qlib_update.sh（Docker 内执行 Dolt 同步 + dump_bin）→ ~/.qlib/qlib_data/cn_data
  2. 导出 daily_pv.h5
  3. 运行因子合成 → combined_factors_df.parquet
  4. 提取 norm_params.pkl（1.8 KB）
  5. 验证所有 22 个参数非 NaN

用法：
    TUSHARE=<token> python scripts/build_full_history.py

时间：约 30-60 分钟（取决于网络和 Docker 性能）
产出：weights/norm_params.pkl（永久有效，除非重新训练模型）
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: str, **kwargs) -> int:
    """Run a shell command, stream output."""
    print(f"\n{'='*60}")
    print(f"[RUN] {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), **kwargs)
    if result.returncode != 0:
        print(f"[FAIL] Exit code {result.returncode}", file=sys.stderr)
    return result.returncode


def main() -> int:
    # Ensure Docker Desktop CLI is in PATH (macOS)
    docker_bin = "/Applications/Docker.app/Contents/Resources/bin"
    if Path(docker_bin).exists() and docker_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{docker_bin}:{os.environ.get('PATH', '')}"

    token = os.getenv("TUSHARE") or os.getenv("TS_TOKEN")
    if not token:
        print("[ERROR] 请设置 TUSHARE 或 TS_TOKEN 环境变量", file=sys.stderr)
        return 1

    cn_data = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    norm_out = REPO_ROOT / "weights" / "norm_params.pkl"

    # ── 步骤 1：用 Docker/Dolt 下载全量数据 ──────────────────────
    print("\n[1/5] 运行 qlib_update.sh（Docker/Dolt 全量下载）...")
    rc = run(f"TUSHARE={token} bash fullstackautoquant/data/qlib_update.sh")
    if rc != 0:
        print("[ERROR] qlib_update.sh 失败。请检查 Docker 是否运行。", file=sys.stderr)
        return 1

    # 验证数据完整性
    cal_file = cn_data / "calendars" / "day.txt"
    inst_file = cn_data / "instruments" / "csi300.txt"
    if not cal_file.exists() or not inst_file.exists():
        print(f"[ERROR] 数据不完整：缺少 {cal_file} 或 {inst_file}", file=sys.stderr)
        return 1

    cal_lines = cal_file.read_text().strip().split("\n")
    inst_lines = inst_file.read_text().strip().split("\n")
    first_date = cal_lines[0].strip() if cal_lines else "?"
    last_date = cal_lines[-1].strip() if cal_lines else "?"
    print(f"  日历范围: {first_date} ~ {last_date} ({len(cal_lines)} 交易日)")
    print(f"  instruments/csi300.txt: {len(inst_lines)} 只股票")

    # 检查 instruments 是否覆盖训练窗口
    first_inst_date = inst_lines[0].split("\t")[1] if inst_lines else "9999"
    if first_inst_date > "2005-02-01":
        print(f"  [INFO] 最早成分股日期: {first_inst_date}")

    # ── 步骤 2：导出 daily_pv.h5 ──────────────────────────────
    print("\n[2/5] 导出 daily_pv.h5...")
    rc = run(
        f"python fullstackautoquant/data/export_daily_pv.py "
        f"--provider_uri {cn_data}"
    )
    if rc != 0:
        print("[ERROR] export_daily_pv.py 失败", file=sys.stderr)
        return 1

    # ── 步骤 3：因子合成 ─────────────────────────────────────
    print("\n[3/5] 运行因子合成...")
    rc = run(
        f"python fullstackautoquant/data/factor_synthesis.py "
        f"--workspace fullstackautoquant/model "
        f"--provider_uri {cn_data}"
    )
    if rc != 0:
        print("[ERROR] factor_synthesis.py 失败", file=sys.stderr)
        return 1

    # ── 步骤 4：提取 norm_params.pkl ──────────────────────────
    print("\n[4/5] 提取 norm_params.pkl...")
    rc = run(
        f"python scripts/extract_norm_cache.py "
        f"--provider_uri {cn_data} "
        f"--out {norm_out}"
    )
    if rc != 0:
        print("[ERROR] extract_norm_cache.py 失败", file=sys.stderr)
        return 1

    # ── 步骤 5：验证 ─────────────────────────────────────────
    print("\n[5/5] 验证 norm_params.pkl...")
    import pickle

    import numpy as np

    params = pickle.load(open(norm_out, "rb"))
    median = params["median"]
    std = params["std"]

    has_nan = np.isnan(median).any() or np.isnan(std).any()
    if has_nan:
        print("[FAIL] norm_params.pkl 仍然包含 NaN！", file=sys.stderr)
        print(f"  median NaN count: {np.isnan(median).sum()}/{len(median)}")
        print(f"  std NaN count: {np.isnan(std).sum()}/{len(std)}")
        return 1

    print(f"\n{'='*60}")
    print(f"  ✅ norm_params.pkl 验证通过")
    print(f"  特征数: {len(median)}")
    print(f"  拟合窗口: [{params.get('fit_start','?')}, {params.get('fit_end','?')}]")
    print(f"  median 范围: [{median.min():.6f}, {median.max():.6f}]")
    print(f"  std 范围: [{std.min():.6f}, {std.max():.6f}]")
    print(f"  文件大小: {norm_out.stat().st_size:,} bytes")
    print(f"  路径: {norm_out}")
    print(f"{'='*60}")
    print("\n此文件应与模型权重一起提交到版本控制。")
    print("之后每日推理不再需要全量 Qlib 数据。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
