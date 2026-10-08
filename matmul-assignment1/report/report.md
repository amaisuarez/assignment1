---
title: "Basic Dense Matrix Multiplication in Python, Java and C: Effect of Matrix Size on Runtime and Memory"
subtitle: "Big Data – Individual Assignment 1"
author: "[YOUR NAME]"
date: "October 2026"
---

<!-- Build the PDF:  pandoc report/report.md -o Assignment01_Surname_Report.pdf --resource-path=report
     Replace every [TODO] with your own text and numbers from report/tables. -->

# Abstract

We study how the matrix size *n* affects the runtime and peak memory of the same i-j-k
triple-loop dense matrix multiplication implemented in Python, Java and C. All
implementations read identical float64 inputs, produce bitwise-identical results and are
timed with the same kernel-only boundary on a single machine ([TODO CPU]). [TODO main
result: e.g. growth exponent, Python/C and Java/C ratios at the largest common n, largest
feasible n per language]. Conclusions are limited to these implementations, settings and
hardware.

# 1. Introduction and objective

Dense matrix multiplication is a basic kernel of data science workloads and has a simple
O(n³) algorithm, which makes it a convenient case for studying how language runtimes and
the memory hierarchy affect performance.

**Research question.** How does *n* affect runtime and memory for the three basic
implementations under the documented conditions?

**Hypotheses (predictions, not results).**

* H1 – Runtime of all three grows approximately as n³ (fitted exponent close to 3).
* H2 – C and Java (JIT-compiled to native code) are within a small factor of each other,
  while CPython is one to two orders of magnitude slower because each inner iteration is
  interpreted bytecode operating on boxed float objects.
* H3 – Time per inner iteration (t/n³) increases once B no longer fits in cache, because
  the inner loop reads B column-wise with a stride of 8n bytes.
* H4 – Peak memory grows as n² for all three, with a larger constant for Python (boxed
  floats) and a larger fixed offset for Java (JVM).

# 2. Background

The classical algorithm computes each of the n² output entries as a dot product of length
n, i.e. n³ multiply-add pairs (2n³ floating-point operations). Its running time on real
hardware also depends on memory access patterns: in the i-j-k order, A is read along rows
(unit stride) but B along columns (stride n), which causes poor spatial locality and cache
misses for large n [1], [2]. High-performance libraries avoid this with blocking and
packing [3], which is deliberately **not** used here.

CPython executes bytecode in an interpreter and represents each float as a heap object, so
the per-operation overhead is large. Java bytecode is compiled at run time by the HotSpot
JIT, so warm-up must be separated from steady-state measurements [4]. C is compiled
ahead of time.

The floating-point error of a length-n dot product is bounded by γₙ·|a|ᵀ|b| with
γₙ = nu/(1−nu) and u = 2⁻⁵³ [5]; this bound is used for the correctness tolerance.

# 3. Methodology

## 3.1 Implementations

All three implement exactly the same loop nest:

```
for i in 0..n-1:
    for j in 0..n-1:
        s = 0
        for k in 0..n-1: s += A[i*n+k] * B[k*n+j]
        C[i*n+j] = s
```

| | Python | Java | C |
|---|---|---|---|
| Storage | flat list of Python `float` objects (pointers to boxed doubles) | `double[]`, contiguous | `double*`, contiguous (`malloc`) |
| Version | [TODO from environment.json] | [TODO] | GCC [TODO] |
| Settings | CPython, no NumPy in kernel | `-XX:+UseSerialGC -Xms256m -Xmx4g`, 3 warm-ups | `-O2 -std=c11 -ffp-contract=off`, 1 warm-up |

Unavoidable difference: the Python list stores pointers to separate float objects instead
of contiguous doubles. `-ffp-contract=off` prevents fused multiply-add in C so the three
programs perform the same IEEE-754 operations in the same order (Java ≥ 17 is strict
IEEE-754 and does not fuse automatically).

## 3.2 Inputs

Square float64 matrices with entries uniform in [−1, 1), generated once by
`scripts/gen_matrices.py` (Python `random.Random`, string seed `20262027:n:A|B`) and stored as
little-endian binary files read by all languages. SHA-256 hashes are in `inputs/manifest.csv`.
Sizes: [TODO copy from config.json], chosen to include powers of two and intermediate values.

## 3.3 Environment

[TODO table from results/run_*/environment.json: CPU model, physical cores / logical
processors, RAM, OS and kernel, Python, Java, GCC versions, CPU governor, power state.]
All kernels are single-threaded. The JVM runs additional internal threads (JIT compiler,
serial GC); CPython runs a single thread.

## 3.4 Correctness

`tests/validate.py` checks (i) exact cases with small integers (hand-computed 2×2, 1×1,
A·I = A, I·B = B, A·0 = 0, random integer matrices up to n = 17), which must match
exactly; and (ii) random float64 matrices (n = 2 … 128) against NumPy as a trusted reference,
accepting |C − C_ref| ≤ 2γₙ(|A||B|) componentwise. [TODO: result – all passed; the three
languages produced bitwise-identical outputs.] During benchmarks the checksum of C is printed
after every repetition; checksums were identical across languages for every n
(`report/tables/checksums.md`).

## 3.5 Measurement protocol

* **Timed region:** kernel only. Input loading, allocation of C, checksum and output are
  outside; C needs no zero-initialisation because every entry is written once.
* **Timers:** monotonic nanosecond timers (`perf_counter_ns`, `System.nanoTime`,
  `clock_gettime(CLOCK_MONOTONIC)`).
* **Batches:** for small n, each sample times `max(1, round(2·10⁷/n³))` consecutive
  multiplications (same batch for all languages); reported values are per multiplication.
* **Repetitions:** one process per (language, n) runs W warm-ups (Python 1, Java 3, C 1) and
  5 measured repetitions. Warm-ups are kept in the raw CSV and excluded from statistics.
  Language order is randomised per size (fixed seed).
* **Statistics:** median and interquartile range (IQR); exponent b from least-squares fit
  of log t = a + b log n for n ≥ 128.
* **Memory:** peak resident set size of the whole process (`VmHWM` from
  `/proc/self/status`), which includes the runtime (interpreter/JVM); for Java the used heap
  is also recorded. Theoretical storage of the three arrays is 24n² bytes.
* **Budgets (declared before running):** 60 s per repetition (process limit (W+5)·60 s + 30 s)
  and 4 GB of memory. After a timeout, larger sizes are not attempted for that language.

# 4. Results

All raw data: `results/[TODO run folder]/`. Figures and tables are generated by
`scripts/analyze.py`.

**Table 1.** Time per multiplication (median and IQR of 5 repetitions), cost per inner
iteration, throughput, ratio to C and peak RSS.

[TODO paste report/tables/summary.md]

**Table 2.** Empirical growth exponent (log-log least squares, n ≥ 128).

[TODO paste report/tables/exponent_fit.md]

**Table 3.** Largest feasible size and failures under the declared budgets.

[TODO paste report/tables/limits.md]

![Runtime versus n (log-log). Points: median; bars: IQR; dashed: n³ reference.](figures/runtime_vs_n.png)

![Median time divided by n³: a flat line means purely cubic growth.](figures/ns_per_iteration.png)

![Peak resident set size of each process versus n, with the theoretical array storage 24n².](figures/peak_memory.png)

![Median runtime of Python and Java relative to C.](figures/slowdown_vs_c.png)

# 5. Discussion

[TODO – answer the research question with your evidence. Points to address:]

* **Growth (H1).** Compare fitted exponents with 3. Values above 3 are explained by the
  rising cost per iteration (Fig. 2), not by a different algorithm.
* **Language effects (H2).** Size of the Python/C and Java/C ratios; why Java approaches C
  after warm-up (JIT); why Python's cost per iteration (~tens of ns) reflects interpretation
  and boxed floats. Explain any small-n differences (batching, JIT warm-up, timer resolution).
* **Memory hierarchy (H3).** Where t/n³ jumps (compare n with cache sizes from `lscpu`):
  the column-wise access to B. Note any anomaly at powers of two (e.g. n = 512, 1024),
  consistent with cache-set conflicts caused by a stride of 8n bytes; when C and Java
  converge, the kernel is memory-bound rather than instruction-bound.
* **Memory (H4).** Fixed runtime offset (JVM ≈ 40 MB, CPython ≈ 10 MB) versus n²
  growth; Python's bytes per element compared with 8 bytes.
* **Variability.** Comment on IQR/median; any outliers (they are kept, not removed).
* **Limitations.** One machine; single-threaded; one process per configuration; laptop
  frequency scaling/thermal effects; memory metric includes runtime; results are not a
  general claim that one language is faster.

# 6. Conclusions

[TODO – what the study establishes within its scope (hardware, settings, sizes), and what
remains open, e.g. effect of loop order (i-k-j), blocking and sparse storage (Assignment 2),
and multithreading (Assignment 3).]

# Use of generative AI

[TODO – write honestly. Example: "Claude (Anthropic) was used to help plan the experimental
design, write the benchmarking/analysis scripts and draft the structure of this report. I
reviewed and understand all code, ran all experiments myself on my own machine, and wrote the
interpretation of the results. All measurements come from my runs; all references were checked."]

# References

[1] J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. Morgan Kaufmann, 2017.

[2] U. Drepper, "What every programmer should know about memory," Red Hat, Inc., 2007.

[3] K. Goto and R. A. van de Geijn, "Anatomy of high-performance matrix multiplication," *ACM Trans. Math. Softw.*, vol. 34, no. 3, 2008.

[4] A. Georges, D. Buytaert, and L. Eeckhout, "Statistically rigorous Java performance evaluation," in *Proc. OOPSLA*, 2007, pp. 57–76.

[5] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002.

Reproducibility materials: GitHub repository [TODO URL], commit [TODO hash].
