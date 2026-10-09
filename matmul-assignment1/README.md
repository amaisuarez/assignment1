# Matrix multiplication in Python, Java and C

**Big Data – Individual Assignment 1** · Amai Suárez Navarro · Grado en Ciencia e Ingeniería de Datos (ULPGC), 2026-2027

This repo has everything behind my first Big Data assignment. I wrote the same basic matrix
multiplication (the classic three nested loops) in Python, Java and C, and measured how time
and memory change as the matrices grow.

The question I wanted to answer:

> How does the matrix size *n* affect the runtime and memory of the same triple-loop
> algorithm in Python, Java and C, on one machine with fixed settings?

The full write-up is in `Assignment01_SuarezNavarro_Report.pdf`.

---

## What I found (short version)

All numbers come from my laptop, an Intel Core i5-5350U MacBook Air with 8 GB of RAM.

- **Python** is very slow for this kind of code: about 110–145× slower than C on small
  matrices and 52–73× slower from n = 256 to 512. It couldn't finish n = 768 within my
  limit of 60 s per run (each run took about 79 s).
- **Java** ends up very close to C. It's 1.3–1.8× slower on small matrices, and for
  n ≥ 1024 it takes exactly the same time as C.
- **The growth isn't a clean n³.** The algorithm always does n³ operations, but each
  operation gets more expensive as the matrices stop fitting in the CPU caches. In C each
  inner iteration goes from about 0.7 ns to about 11 ns. That's why the fitted exponent
  comes out close to 4 over the whole range, but close to 3 inside each range of sizes.
- **Memory** was never the problem: C uses about 8 bytes per element, Java about 17
  (plus ~36 MB for the JVM), and Python about 40 because every number is a separate object.

The three programs give **exactly the same result bit for bit**, so I'm confident they all
do the same computation.

---

## What's in the repo

```
c/matmul.c                 C version
java/MatMul.java           Java version
python/matmul.py           Python version (plain Python, no NumPy)

scripts/gen_matrices.py    creates the input matrices (the same files for all 3 languages)
scripts/run_benchmarks.py  runs the experiment
scripts/collect_env.py     saves info about the machine (CPU, RAM, versions...)
scripts/analyze.py         builds the tables and plots from the raw data
tests/validate.py          checks that the 3 versions are correct

config.json                settings of the real study (sizes, repetitions, limits)
config_quick.json          quick test with small sizes, just to check everything works
build.sh                   compiles the C and Java code

inputs/manifest.csv        seeds and SHA-256 of each input matrix
results/                   raw data from every run, plus the validation report
report/                    report source, tables and figures
```

The study used for the report is `results/run_20261008-202121/`. The other folder in
`results/` is the quick test I ran first to check the pipeline; it's not used in the report.

---

## What you need

- macOS or Linux (on Windows, use WSL)
- Python 3.9 or newer
- Java JDK 17 or newer (I used OpenJDK 21)
- A C compiler (`gcc`; on a Mac this is actually clang, which works the same here)

The Python version of the algorithm doesn't use any libraries. NumPy is only used to check
results in the tests, and Matplotlib to draw the plots:

```bash
pip install -r requirements.txt
```

---

## How to run it

**1. Compile C and Java**

```bash
bash build.sh
```

**2. Check that the three versions are correct**

```bash
python3 tests/validate.py
```

It should finish with `ALL TESTS PASSED`. It tests small cases with known answers (identity
matrix, zero matrix, a 2×2 done by hand) and compares random matrices with NumPy.

**3. (Optional) Quick test, about 1 minute**

```bash
python3 scripts/run_benchmarks.py --config config_quick.json
```

**4. The real study**

This takes a while (around 40–60 minutes on my laptop). Keep the computer plugged in and
don't use it meanwhile, otherwise the measurements get noisy. On a Mac, `caffeinate` stops it
from going to sleep:

```bash
caffeinate -i python3 scripts/run_benchmarks.py    # on Linux, just drop "caffeinate -i"
```

At the end it prints the name of the new folder, something like `results/run_YYYYMMDD-HHMMSS`.

**5. Tables and plots**

```bash
python3 scripts/analyze.py results/run_YYYYMMDD-HHMMSS
```

This writes the tables to `report/tables/` and the plots to `report/figures/`.

You don't need to download any data. The input matrices are generated automatically in
`data/` (git ignores that folder because it's large) and they always come out the same thanks
to fixed seeds. `inputs/manifest.csv` holds their SHA-256 hashes so you can check it.

---

## How I made the comparison fair

| | What I did |
|---|---|
| Algorithm | Same i-j-k triple loop in the three languages |
| Numbers | float64 everywhere. In C I compile with `-ffp-contract=off` so it doesn't merge multiply and add into one instruction, which would change the rounding |
| Inputs | The three read the same binary files. Values uniform in [-1, 1), seed `20262027` |
| What I time | Only the multiplication loops. Reading files, allocating memory and checking the result are outside the timer |
| Timer | Monotonic nanosecond timer in each language |
| Small matrices | They take microseconds, so I time a batch of them and divide |
| Repetitions | 5 measured runs per case, plus warm-up runs that don't count (3 in Java because of the JIT compiler, 1 in Python and C) |
| Statistics | Median and interquartile range (IQR) |
| Memory | Peak RSS of each process (on macOS, `ru_maxrss`) |
| Limits (set beforehand) | 60 s per run and 4 GB of memory. If a size times out, it's recorded as a timeout and I don't try bigger sizes for that language |

Settings that stayed fixed:

- **C:** `gcc -O2 -std=c11 -ffp-contract=off`
- **Java:** `-XX:+UseSerialGC -Xms256m -Xmx4g`, compiled with `--release 17`
- **Python:** CPython 3.13.5 (Anaconda)

---

## Raw data format

- `results/run_*/raw_measurements.csv`: one row per run, with method, n, numeric type,
  threads, batch, whether it was a warm-up or measured run, time in ns, time per
  multiplication and the checksum.
- `results/run_*/runs.csv`: one row per (language, n), with whether it finished
  (`ok`, `timeout`...), peak memory and the time limit used.
- `results/run_*/environment.json`: the machine and version details.

---

## Limitations

Everything was measured on one laptop. I couldn't control Turbo Boost or pin the process to
one core (macOS doesn't allow it). The memory figure includes the runtime of each language
(JVM, Python interpreter), so it isn't exactly the same kind of number for all three. The
results describe these three simple implementations on this machine; they're not a general
claim about which language is faster.

---

## Use of AI

I used Claude (Anthropic) to help with the design of the experiment, and some explanations about the topic just for me to learn about it. I ran all the
experiments on my own computer, and every number here comes from those runs.
