#!/usr/bin/env python3
"""Correctness validation of the three implementations (run ./build.sh first).

Cases
  * exact cases (small integers, exactly representable, so results must be EXACT):
    hand-computed 2x2, 1x1, identity (A x I = A, I x B = B), zero matrix.
  * random cases vs. NumPy (trusted reference, NOT one of the studied methods).

Tolerance for random cases (float64)
  Standard error bound for a length-n dot product computed in floating point
  (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., §3.1):
      |fl(C) - C| <= gamma_n * (|A| |B|),   gamma_n = n*u / (1 - n*u),  u = 2^-53.
  Both our result and NumPy's satisfy it, so we accept componentwise
      |C_ours - C_numpy| <= 2 * gamma_n * (|A| |B|).
  This scales with n and with the magnitudes involved, which a fixed rtol/atol does not.

Also reported: whether the three languages give BITWISE identical results
(expected, since they perform the same IEEE-754 operations in the same order
and FMA contraction is disabled).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
U = 2.0 ** -53

METHODS = {
    "python": [sys.executable, str(ROOT / "python" / "matmul.py")],
    "java": ["java", "-cp", str(ROOT / "java" / "build"), "MatMul"],
    "c": [str(ROOT / "c" / "build" / "matmul")],
}


def write(path, M):
    np.ascontiguousarray(M, dtype="<f8").tofile(path)


def run(method, A, B, tmp):
    n = A.shape[0]
    pa, pb, pc = tmp / "A.bin", tmp / "B.bin", tmp / f"C_{method}.bin"
    write(pa, A)
    write(pb, B)
    cmd = METHODS[method] + ["--n", str(n), "--a", str(pa), "--b", str(pb),
                             "--warmup", "0", "--reps", "1", "--batch", "1", "--out", str(pc)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    C = np.fromfile(pc, dtype="<f8").reshape(n, n)
    rep_lines = [l for l in out.splitlines() if not l.startswith("#")]
    reported = float(rep_lines[-1].split(",")[-1])
    checksum = 0.0
    for x in C.ravel():          # same sequential order as the implementations
        checksum += float(x)
    assert reported == checksum, f"{method}: printed checksum does not match output"
    return C


def exact_cases(rng):
    A2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    B2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    yield "2x2 hand-computed", A2, B2, np.array([[19.0, 22.0], [43.0, 50.0]])
    yield "1x1", np.array([[3.0]]), np.array([[-2.0]]), np.array([[-6.0]])
    for n in (3, 8, 17):
        Ai = rng.integers(-9, 10, size=(n, n)).astype(float)
        I = np.eye(n)
        yield f"A x I (n={n})", Ai, I, Ai
        yield f"I x B (n={n})", I, Ai, Ai
        yield f"A x 0 (n={n})", Ai, np.zeros((n, n)), np.zeros((n, n))
        Bi = rng.integers(-9, 10, size=(n, n)).astype(float)
        yield f"small integers (n={n})", Ai, Bi, Ai @ Bi   # exact: |values| << 2^53


def main():
    rng = np.random.default_rng(12345)
    failures = 0
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        log("== Exact cases (must match exactly) ==")
        for name, A, B, expected in exact_cases(rng):
            for m in METHODS:
                C = run(m, A, B, tmp)
                ok = np.array_equal(C, expected)
                failures += not ok
                log(f"  [{'PASS' if ok else 'FAIL'}] {m:6s} {name}")

        log("== Random cases vs NumPy (componentwise bound 2*gamma_n*|A||B|) ==")
        for n in (2, 5, 16, 33, 64, 100, 128):
            A = rng.uniform(-1, 1, (n, n))
            B = rng.uniform(-1, 1, (n, n))
            ref = A @ B
            gamma = n * U / (1 - n * U)
            bound = 2 * gamma * (np.abs(A) @ np.abs(B))
            results = {}
            for m in METHODS:
                C = run(m, A, B, tmp)
                results[m] = C
                err = np.abs(C - ref)
                ok = bool(np.all(err <= bound))
                failures += not ok
                log(f"  [{'PASS' if ok else 'FAIL'}] {m:6s} n={n:4d}  max|err|={err.max():.3e}  "
                    f"max(err/bound)={np.max(err / np.where(bound > 0, bound, 1)):.3f}")
            same = all(np.array_equal(results["c"], results[m]) for m in METHODS)
            log(f"         bitwise identical across languages: {same}")

    log(f"== {'ALL TESTS PASSED' if failures == 0 else f'{failures} FAILURE(S)'} ==")
    out = ROOT / "results" / "validation_report.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
