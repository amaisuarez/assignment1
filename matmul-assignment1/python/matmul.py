#!/usr/bin/env python3
"""Basic dense matrix multiplication, triple loop (i-j-k), pure Python.

Storage: flat row-major Python list of floats (element (i, j) at index i*n + j).
Note: a Python list stores pointers to boxed float objects, unlike the
contiguous double arrays used in C and Java. This is an unavoidable
language-level difference and is discussed in the report.

Output format (stdout), one line per repetition:
    kind,index,elapsed_ns,batch,checksum
kind is "warmup" or "measured". elapsed_ns covers `batch` multiplications.
"""
import argparse
import sys
import time
from array import array


def load(path, n):
    a = array("d")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    if len(a) != n * n:
        sys.exit(f"{path}: expected {n * n} values, found {len(a)}")
    return a.tolist()


def matmul(A, B, C, n):
    """C = A x B. O(n^3): every C element written once, no zero-init needed."""
    for i in range(n):
        row = i * n
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[row + k] * B[k * n + j]
            C[row + j] = s


def checksum(C):
    s = 0.0
    for x in C:          # sequential sum, same order as C and Java
        s += x
    return s


def peak_rss_bytes():
    """Peak resident set size of this process (Linux VmHWM); -1 if unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def save(path, C):
    a = array("d", C)
    if sys.byteorder != "little":
        a.byteswap()
    with open(path, "wb") as f:
        a.tofile(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    n = args.n
    A = load(args.a, n)
    B = load(args.b, n)
    C = [0.0] * (n * n)          # allocated once, outside the timed region

    for kind, count in (("warmup", args.warmup), ("measured", args.reps)):
        for r in range(count):
            t0 = time.perf_counter_ns()          # monotonic, high resolution
            for _ in range(args.batch):
                matmul(A, B, C, n)
            t1 = time.perf_counter_ns()
            # result consumed after the timed region
            print(f"{kind},{r},{t1 - t0},{args.batch},{checksum(C)!r}", flush=True)

    if args.out:
        save(args.out, C)
    print(f"#peak_rss_bytes,{peak_rss_bytes()}", flush=True)


if __name__ == "__main__":
    main()
