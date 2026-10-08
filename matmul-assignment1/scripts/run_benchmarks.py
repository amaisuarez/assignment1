#!/usr/bin/env python3
"""Benchmark harness for Assignment 1.

For every size n (ascending) and every method, it launches ONE process that
loads the shared inputs, runs W warm-up and R measured repetitions and prints
one line per repetition. Each repetition times `batch` multiplications, where
    batch = max(1, round(batch_target_inner_iterations / n^3))
so that tiny sizes are measured as batches (reported per multiplication).
The batch is identical for all methods at a given n.

Timing boundary (kernel): only the triple loop. Input loading, allocation of C,
the checksum and file output are OUTSIDE the timed region.

Memory: peak resident set size of that process. Each program reports its own
VmHWM from /proc/self/status (Linux; exact, reset at exec). Where /proc is not
available (macOS) the harness falls back to ru_maxrss from os.wait4(), which on
some systems also counts memory of the forked parent before exec (documented
as metric "ru_maxrss_wait4"). Java also reports used heap after the run.

Budgets: a process is killed after (W+R)*per_repetition_budget_s + startup
allowance -> status "timeout"; larger sizes are then not attempted for that
method. If the estimated array storage exceeds the memory budget the run is
skipped -> "skipped_memory_budget". No time is ever invented for these.

Output: results/run_<timestamp>/{raw_measurements.csv, runs.csv,
        environment.json, config.json, stderr/}
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_env  # noqa: E402
import gen_matrices  # noqa: E402

RAW_FIELDS = ["run_id", "method", "language", "n", "dtype", "threads", "batch", "kind",
              "rep_index", "elapsed_ns", "time_per_mult_s", "checksum", "status"]
RUN_FIELDS = ["run_id", "method", "n", "status", "exit_code", "warmups", "repetitions", "batch",
              "peak_rss_bytes", "memory_metric", "java_heap_used_bytes", "array_storage_bytes", "estimated_bytes", "wall_time_s",
              "timeout_s", "order_in_size", "command"]


def command(method, cfg, n, pa, pb, warmup, reps, batch):
    args = ["--n", str(n), "--a", str(pa), "--b", str(pb),
            "--warmup", str(warmup), "--reps", str(reps), "--batch", str(batch)]
    if method == "python":
        cmd = [sys.executable, str(ROOT / "python" / "matmul.py")] + args
    elif method == "java":
        cmd = ["java"] + cfg["java_opts"] + ["-cp", str(ROOT / "java" / "build"), "MatMul"] + args
    elif method == "c":
        cmd = [str(ROOT / "c" / "build" / "matmul")] + args
    else:
        raise ValueError(method)
    if cfg.get("pin_to_cpu") is not None and sys.platform.startswith("linux"):
        cmd = ["taskset", "-c", str(cfg["pin_to_cpu"])] + cmd
    return cmd


def run_process(cmd, timeout_s, stderr_path):
    """Run cmd, return (stdout, exit_code, timed_out, peak_rss_bytes, wall_s)."""
    with open(stderr_path, "w") as err:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            p.kill()

        timer = threading.Timer(timeout_s, kill)
        timer.start()
        out = p.stdout.read()              # until the process exits (or is killed)
        _, status, usage = os.wait4(p.pid, 0)   # rusage of THIS child only
        wall = time.monotonic() - t0
        timer.cancel()
        p.returncode = os.waitstatus_to_exitcode(status)
        p.stdout.close()
    scale = 1 if sys.platform == "darwin" else 1024     # Linux reports KiB, macOS bytes
    return out, p.returncode, timed_out.is_set(), usage.ru_maxrss * scale, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--only", nargs="*", help="subset of methods")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    methods = args.only or cfg["methods"]

    for exe in [ROOT / "c" / "build" / "matmul", ROOT / "java" / "build" / "MatMul.class"]:
        if not exe.exists():
            sys.exit("Build first: ./build.sh")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / f"run_{run_id}"
    (out_dir / "stderr").mkdir(parents=True)
    shutil.copy(args.config, out_dir / "config.json")
    (out_dir / "environment.json").write_text(json.dumps(collect_env.collect(cfg), indent=2))

    raw_f = open(out_dir / "raw_measurements.csv", "w", newline="")
    runs_f = open(out_dir / "runs.csv", "w", newline="")
    raw = csv.DictWriter(raw_f, RAW_FIELDS, lineterminator="\n")
    runs = csv.DictWriter(runs_f, RUN_FIELDS, lineterminator="\n")
    raw.writeheader()
    runs.writeheader()

    stopped = {}                      # method -> reason larger sizes are not attempted
    order_rng = random.Random(cfg["seed"])
    R = cfg["repetitions"]

    for n in cfg["sizes"]:
        pa, pb = gen_matrices.ensure(n, cfg["seed"])
        batch = max(1, round(cfg["batch_target_inner_iterations"] / n ** 3))
        order = list(methods)
        order_rng.shuffle(order)      # randomised method order per size (limits drift bias)
        for pos, m in enumerate(order):
            W = cfg["warmups"][m]
            storage = 3 * 8 * n * n
            est = 3 * cfg["bytes_per_element_estimate"][m] * n * n
            timeout = (W + R) * cfg["per_repetition_budget_s"] + cfg["process_startup_allowance_s"]
            base = {"run_id": run_id, "method": m, "n": n, "warmups": W, "repetitions": R,
                    "batch": batch, "array_storage_bytes": storage, "estimated_bytes": est,
                    "timeout_s": timeout, "order_in_size": pos}
            if m in stopped:
                runs.writerow({**base, "status": stopped[m]})
                continue
            if est > cfg["memory_budget_bytes"]:
                runs.writerow({**base, "status": "skipped_memory_budget"})
                stopped[m] = "not_attempted_after_memory_budget"
                continue

            cmd = command(m, cfg, n, pa, pb, W, R, batch)
            print(f"[n={n:5d}] {m:6s} batch={batch:<5d} W={W} R={R} ...", end=" ", flush=True)
            out, code, timed_out, rss, wall = run_process(
                cmd, timeout, out_dir / "stderr" / f"{m}_n{n}.txt")
            status = "timeout" if timed_out else ("ok" if code == 0 else "error")

            extra = {"peak_rss_bytes": rss, "memory_metric": "ru_maxrss_wait4",
                     "java_heap_used_bytes": ""}
            for line in out.strip().splitlines():
                if line.startswith("#"):
                    key, value = line[1:].split(",")
                    if key == "peak_rss_bytes" and int(value) > 0:
                        extra.update(peak_rss_bytes=int(value), memory_metric="VmHWM_proc_self_status")
                    elif key == "java_heap_used_bytes":
                        extra["java_heap_used_bytes"] = int(value)
                    continue
                kind, idx, ns, b, chk = line.split(",")
                raw.writerow({"run_id": run_id, "method": m, "language": m, "n": n,
                              "dtype": "float64", "threads": 1, "batch": int(b), "kind": kind,
                              "rep_index": int(idx), "elapsed_ns": int(ns),
                              "time_per_mult_s": int(ns) / 1e9 / int(b), "checksum": chk,
                              "status": status})
            print(f"{status} ({wall:.1f}s, peak RSS {extra['peak_rss_bytes'] / 2**20:.1f} MiB)")
            runs.writerow({**base, **extra, "status": status, "exit_code": code,
                           "wall_time_s": round(wall, 3),
                           "command": " ".join(cmd)})
            raw_f.flush()
            runs_f.flush()
            if status != "ok":
                stopped[m] = f"not_attempted_after_{status}"

    raw_f.close()
    runs_f.close()
    print(f"\nResults written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
