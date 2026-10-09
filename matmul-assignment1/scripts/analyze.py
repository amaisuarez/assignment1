#!/usr/bin/env python3
"""Generate tables and figures for the report from one benchmark run.

Usage: python3 scripts/analyze.py [results/run_XXXX]   (default: latest run)

Statistics (per method and n, MEASURED repetitions only; warm-ups are kept in
the raw CSV but excluded here): median, Q1, Q3, IQR, min, max, count.
Derived: ns per inner iteration = median / n^3, GFLOP/s = 2 n^3 / median,
slowdown relative to C, and an empirical exponent b from a least-squares fit
of log(median) = a + b log(n) for n >= FIT_MIN_N.

Outputs: report/tables/*.csv|*.md and report/figures/*.png
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIT_MIN_N = 128
# Size regimes for a piecewise exponent fit (inclusive bounds), chosen from the
# normalised-cost plot: small (inputs fit in cache), intermediate, large.
FIT_RANGES = [(16, 192), (256, 768), (1024, 4096)]
STYLE = {"python": ("tab:blue", "o"), "java": ("tab:orange", "s"), "c": ("tab:green", "^")}
LABEL = {"python": "Python", "java": "Java", "c": "C"}


def latest_run():
    runs = sorted((ROOT / "results").glob("run_*"))
    if not runs:
        sys.exit("No runs found in results/")
    return runs[-1]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def fmt_time(s):
    if s < 1e-3:
        return f"{s * 1e6:.2f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    raw = load_csv(run / "raw_measurements.csv")
    runs = load_csv(run / "runs.csv")
    env = json.loads((run / "environment.json").read_text())
    tables = ROOT / "report" / "tables"
    figs = ROOT / "report" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    samples = defaultdict(list)
    checksums = defaultdict(set)
    for r in raw:
        if r["kind"] == "measured" and r["status"] == "ok":
            samples[(r["method"], int(r["n"]))].append(float(r["time_per_mult_s"]))
            checksums[int(r["n"])].add(float(r["checksum"]))
    mem = {(r["method"], int(r["n"])): r for r in runs}
    methods = [m for m in ["python", "java", "c"] if any(k[0] == m for k in samples)]
    sizes = sorted({n for _, n in samples})

    # ---- summary table ------------------------------------------------------
    summary = []
    for m in methods:
        for n in sizes:
            x = np.array(samples.get((m, n), []))
            if not len(x):
                continue
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            rss = mem[(m, n)]["peak_rss_bytes"]
            summary.append({
                "method": m, "n": n, "reps": len(x), "batch": mem[(m, n)]["batch"],
                "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                "rel_iqr_pct": 100 * (q3 - q1) / med, "min_s": x.min(), "max_s": x.max(),
                "ns_per_inner_iter": med / n ** 3 * 1e9, "gflops": 2 * n ** 3 / med / 1e9,
                "peak_rss_mib": int(rss) / 2 ** 20 if rss else float("nan"),
                "array_storage_mib": 24 * n * n / 2 ** 20,
            })
    by = {(s["method"], s["n"]): s for s in summary}
    for s in summary:
        c = by.get(("c", s["n"]))
        s["slowdown_vs_c"] = s["median_s"] / c["median_s"] if c else float("nan")

    keys = list(summary[0].keys())
    with open(tables / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(summary)

    pretty = [{"Method": LABEL[s["method"]], "n": s["n"], "Reps": s["reps"],
               "Median": fmt_time(s["median_s"]), "IQR": fmt_time(s["iqr_s"]),
               "IQR/median": f"{s['rel_iqr_pct']:.1f}%",
               "ns/iter": f"{s['ns_per_inner_iter']:.2f}", "GFLOP/s": f"{s['gflops']:.3f}",
               "×C": f"{s['slowdown_vs_c']:.1f}", "Peak RSS (MiB)": f"{s['peak_rss_mib']:.1f}"}
              for s in summary]
    (tables / "summary.md").write_text(md_table(pretty, list(pretty[0].keys())))

    # ---- empirical exponent -------------------------------------------------
    fits = []
    for m in methods:
        pts = [(s["n"], s["median_s"]) for s in summary if s["method"] == m and s["n"] >= FIT_MIN_N]
        if len(pts) >= 3:
            ln = np.log([p[0] for p in pts])
            lt = np.log([p[1] for p in pts])
            b, a = np.polyfit(ln, lt, 1)
            fits.append({"Method": LABEL[m], "fit range": f"{pts[0][0]}–{pts[-1][0]}",
                         "points": len(pts), "exponent b": f"{b:.2f}"})
    if fits:
        (tables / "exponent_fit.md").write_text(md_table(fits, list(fits[0].keys())))

    # ---- piecewise exponent fit ------------------------------------------
    piece = []
    for m in methods:
        row = {"Method": LABEL[m]}
        for lo, hi in FIT_RANGES:
            pts = [(s["n"], s["median_s"]) for s in summary
                   if s["method"] == m and lo <= s["n"] <= hi]
            key = f"n {lo}–{hi if hi < 4096 else 'max'}"
            if len(pts) >= 2:
                b = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]
                row[key] = f"{b:.2f} ({pts[0][0]}–{pts[-1][0]}, {len(pts)} pts)"
            else:
                row[key] = "–"
        piece.append(row)
    (tables / "exponent_piecewise.md").write_text(md_table(piece, list(piece[0].keys())))

    # ---- warm-up effect (first warm-up vs median of measured) ---------------
    warm = defaultdict(list)
    for r in raw:
        if r["kind"] == "warmup" and r["status"] == "ok":
            warm[(r["method"], int(r["n"]))].append(float(r["time_per_mult_s"]))
    wrows = []
    for m in methods:
        for n in sizes:
            if warm.get((m, n)) and (m, n) in by:
                med = by[(m, n)]["median_s"]
                wrows.append({"Method": LABEL[m], "n": n,
                              "first warm-up / median": f"{warm[(m, n)][0] / med:.2f}",
                              "last warm-up / median": f"{warm[(m, n)][-1] / med:.2f}"})
    if wrows:
        (tables / "warmup_effect.md").write_text(md_table(wrows, list(wrows[0].keys())))

    # ---- repetitions completed before a timeout (not used in statistics) ----
    part = defaultdict(list)
    for r in raw:
        if r["status"] == "timeout":
            part[(r["method"], int(r["n"]))].append((r["kind"], float(r["time_per_mult_s"])))
    prow = []
    for (m, n), xs in sorted(part.items()):
        meas = [t for k, t in xs if k == "measured"]
        prow.append({"Method": LABEL[m], "n": n, "warm-ups done": sum(k == "warmup" for k, _ in xs),
                     "measured reps done": len(meas),
                     "time per completed rep [s]": ", ".join(f"{t:.1f}" for t in meas) or "–"})
    if prow:
        (tables / "timeout_partial.md").write_text(
            "Repetitions completed before the process limit was reached. They exceed the "
            "declared 60 s per-repetition budget and are excluded from all statistics.\n\n"
            + md_table(prow, list(prow[0].keys())))

    # ---- limits / statuses --------------------------------------------------
    limits = []
    for m in methods:
        rs = [r for r in runs if r["method"] == m]
        ok = [int(r["n"]) for r in rs if r["status"] == "ok"]
        attempted = [int(r["n"]) for r in rs if r["status"] in ("ok", "timeout", "error")]
        fail = [f"n={r['n']}: {r['status']} (limit {r['timeout_s']} s)" for r in rs
                if r["status"] in ("timeout", "error", "skipped_memory_budget")]
        limits.append({"Method": LABEL[m], "Largest feasible n": max(ok) if ok else "-",
                       "Largest attempted n": max(attempted) if attempted else "-",
                       "Failures": "; ".join(fail) or "none"})
    (tables / "limits.md").write_text(md_table(limits, list(limits[0].keys())))

    same = {n: len(v) == 1 for n, v in checksums.items()}
    (tables / "checksums.md").write_text(
        "Checksums identical across all methods and repetitions for every n: "
        f"**{all(same.values())}**\n\n" +
        md_table([{"n": n, "identical": v} for n, v in sorted(same.items())], ["n", "identical"]))

    # ---- figures ------------------------------------------------------------
    def series(m, key):
        ss = [s for s in summary if s["method"] == m]
        return np.array([s["n"] for s in ss]), np.array([s[key] for s in ss]), ss

    # 1. runtime vs n (log-log) with IQR bars and n^3 reference
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        n, med, ss = series(m, "median_s")
        err = np.array([[s["median_s"] - s["q1_s"] for s in ss], [s["q3_s"] - s["median_s"] for s in ss]])
        col, mk = STYLE[m]
        ax.errorbar(n, med, yerr=err, color=col, marker=mk, capsize=3, label=LABEL[m])
    if "c" in methods:
        n, med, _ = series("c", "median_s")
        ref = med[0] * (n / n[0]) ** 3
        ax.plot(n, ref, "k--", lw=1, label=r"$\propto n^3$ (anchored at C, smallest n)")
    ax.set(xscale="log", yscale="log", xlabel="Matrix size n (n × n)",
           ylabel="Time per multiplication [s] (median, bars = IQR)",
           title="Runtime vs matrix size")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "runtime_vs_n.png", dpi=150)

    # 2. normalised cost: ns per inner-loop iteration
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        n, v, _ = series(m, "ns_per_inner_iter")
        col, mk = STYLE[m]
        ax.plot(n, v, color=col, marker=mk, label=LABEL[m])
    ax.set(xscale="log", yscale="log", xlabel="Matrix size n",
           ylabel="Median time / n³ [ns per inner iteration]",
           title="Normalised cost (flat line = pure cubic growth)")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "ns_per_iteration.png", dpi=150)

    # 3. memory
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        n, v, _ = series(m, "peak_rss_mib")
        col, mk = STYLE[m]
        ax.plot(n, v, color=col, marker=mk, label=f"{LABEL[m]} peak RSS")
    nn = np.array(sizes)
    ax.plot(nn, 24 * nn ** 2 / 2 ** 20, "k--", lw=1, label="3 dense float64 arrays (24 n² B)")
    ax.set(xscale="log", yscale="log", xlabel="Matrix size n", ylabel="Memory [MiB]",
           title="Peak resident set size per process")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "peak_memory.png", dpi=150)

    # 4. slowdown vs C
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        if m == "c":
            continue
        n, v, _ = series(m, "slowdown_vs_c")
        col, mk = STYLE[m]
        ax.plot(n, v, color=col, marker=mk, label=f"{LABEL[m]} / C")
    ax.axhline(1, color="k", lw=.8)
    ax.set(xscale="log", yscale="log", xlabel="Matrix size n",
           ylabel="Median time ratio relative to C", title="Relative slowdown vs C")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "slowdown_vs_c.png", dpi=150)

    cpu = env["cpu"]
    print(f"Run: {run.name} | CPU: {cpu.get('model')} | "
          f"{cpu.get('physical_cores')} cores / {cpu.get('logical_processors')} logical")
    print((tables / "summary.md").read_text())
    print((tables / "exponent_fit.md").read_text() if fits else "")
    print((tables / "limits.md").read_text())
    for extra in ("exponent_piecewise.md", "warmup_effect.md", "timeout_partial.md"):
        if (tables / extra).exists():
            print((tables / extra).read_text())
    print(f"Tables -> {tables.relative_to(ROOT)}  Figures -> {figs.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
