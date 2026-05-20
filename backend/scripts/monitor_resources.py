"""Standalone resource monitor for AugMentor evaluation runs.

Run from the backend/ directory while an evaluation is in progress:

    python scripts/monitor_resources.py

Press CTRL+C to stop. Outputs are saved to backend/resource_logs/.
"""
from __future__ import annotations

import csv
import datetime
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[monitor] matplotlib not found — PNG graph will be skipped.")
    print("[monitor] Install with:  pip install matplotlib")

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_INTERVAL_SEC = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "resource_logs"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_gpu() -> tuple[float, float]:
    """Return (gpu_util_percent, gpu_mem_percent) via nvidia-smi.

    Returns (0.0, 0.0) if nvidia-smi is unavailable or errors.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return 0.0, 0.0
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
        gpu_mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
        return gpu_util, gpu_mem_pct
    except Exception:
        return 0.0, 0.0


def _save_png(rows: list[dict], png_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    elapsed   = [r["elapsed_sec"]    for r in rows]
    cpu       = [r["cpu_percent"]    for r in rows]
    ram       = [r["ram_percent"]    for r in rows]
    gpu_util  = [r["gpu_percent"]    for r in rows]
    gpu_mem   = [r["gpu_mem_percent"] for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("AugMentor Evaluation — Resource Monitor", fontsize=13, fontweight="bold")

    for ax, data, label, color in zip(
        axes,
        [cpu, ram, gpu_util, gpu_mem],
        ["CPU Usage (%)", "RAM Usage (%)", "GPU Usage (%)", "GPU Memory (%)"],
        ["steelblue", "darkorange", "seagreen", "mediumpurple"],
    ):
        ax.plot(elapsed, data, color=color, linewidth=1.2)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.fill_between(elapsed, data, alpha=0.15, color=color)

    axes[-1].set_xlabel("Elapsed time (s)", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"resource_run_{ts}.csv"
    png_path = OUTPUT_DIR / f"resource_run_{ts}.png"

    print("=" * 60)
    print("  AugMentor Resource Monitor")
    print("=" * 60)
    print(f"  Sample interval : {SAMPLE_INTERVAL_SEC}s")
    print(f"  Output CSV      : {csv_path}")
    print(f"  Output PNG      : {png_path}")
    print("  Press CTRL+C to stop.\n")

    # Warm up psutil cpu_percent (first call always returns 0)
    psutil.cpu_percent(interval=None)

    rows: list[dict] = []
    start_time = time.perf_counter()

    columns = ["elapsed_sec", "cpu_percent", "ram_percent", "gpu_percent", "gpu_mem_percent"]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        def _shutdown(sig, frame) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _shutdown)
        if hasattr(signal, "SIGBREAK"):          # Windows CTRL+Break
            signal.signal(signal.SIGBREAK, _shutdown)

        try:
            while True:
                elapsed      = round(time.perf_counter() - start_time, 2)
                cpu_pct      = psutil.cpu_percent(interval=None)
                ram_pct      = psutil.virtual_memory().percent
                gpu_pct, gpu_mem_pct = _query_gpu()

                row = {
                    "elapsed_sec":     elapsed,
                    "cpu_percent":     round(cpu_pct, 1),
                    "ram_percent":     round(ram_pct, 1),
                    "gpu_percent":     round(gpu_pct, 1),
                    "gpu_mem_percent": round(gpu_mem_pct, 1),
                }
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)

                print(
                    f"  [{elapsed:>8.1f}s]  "
                    f"CPU {cpu_pct:5.1f}%  "
                    f"RAM {ram_pct:5.1f}%  "
                    f"GPU {gpu_pct:5.1f}%  "
                    f"GPU-MEM {gpu_mem_pct:5.1f}%",
                    flush=True,
                )

                time.sleep(SAMPLE_INTERVAL_SEC)

        except KeyboardInterrupt:
            pass

    print("\n" + "=" * 60)
    print("  Monitoring stopped.")
    print(f"  Samples collected : {len(rows)}")
    print(f"  CSV saved to      : {csv_path}")

    if rows:
        _save_png(rows, png_path)
        if HAS_MATPLOTLIB:
            print(f"  PNG saved to      : {png_path}")
    else:
        print("  No samples collected — PNG not generated.")

    print("=" * 60)


if __name__ == "__main__":
    main()
