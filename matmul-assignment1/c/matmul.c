/*
 * Basic dense matrix multiplication, triple loop (i-j-k), C11.
 *
 * Storage: contiguous row-major double array (element (i,j) at i*n + j).
 * Build:   gcc -O2 -std=c11 -ffp-contract=off -o c/build/matmul c/matmul.c
 *          (-ffp-contract=off forbids fused multiply-add so the floating-point
 *           operations are the same as in Python and Java.)
 *
 * Output format (stdout), one line per repetition:
 *     kind,index,elapsed_ns,batch,checksum
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int is_little_endian(void) {
    const uint16_t x = 1;
    return *(const uint8_t *)&x == 1;
}

static double *load(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    size_t count = n * n;
    double *m = malloc(count * sizeof(double));
    if (!m) { fprintf(stderr, "malloc failed\n"); exit(2); }
    size_t got = fread(m, sizeof(double), count, f);
    if (got != count || fgetc(f) != EOF) {
        fprintf(stderr, "%s: expected %zu values\n", path, count);
        exit(1);
    }
    fclose(f);
    return m;
}

static void save(const char *path, const double *C, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(C, sizeof(double), n * n, f);
    fclose(f);
}

/* noinline keeps the kernel a separate, inspectable function */
__attribute__((noinline))
static void matmul(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        size_t row = (size_t)i * n;
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[row + k] * B[(size_t)k * n + j];
            C[row + j] = s;
        }
    }
}

static double checksum(const double *C, size_t n) {
    double s = 0.0;
    for (size_t idx = 0; idx < n * n; idx++) s += C[idx];
    return s;
}

/* Peak resident set size of this process (Linux VmHWM); -1 if unavailable. */
static long long peak_rss_bytes(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %lld kB", &kb) == 1) break;
    fclose(f);
    return kb < 0 ? -1 : kb * 1024;
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int n = -1, warmup = 0, reps = 1, batch = 1;
    const char *pa = NULL, *pb = NULL, *out = NULL;
    for (int i = 1; i + 1 < argc; i += 2) {
        if      (!strcmp(argv[i], "--n"))      n = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--a"))      pa = argv[i + 1];
        else if (!strcmp(argv[i], "--b"))      pb = argv[i + 1];
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--reps"))   reps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--batch"))  batch = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--out"))    out = argv[i + 1];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (n <= 0 || !pa || !pb) {
        fprintf(stderr, "usage: matmul --n N --a A.bin --b B.bin [--warmup W] [--reps R] [--batch K] [--out C.bin]\n");
        return 1;
    }
    if (!is_little_endian()) { fprintf(stderr, "big-endian host not supported\n"); return 1; }

    double *A = load(pa, (size_t)n);
    double *B = load(pb, (size_t)n);
    double *C = malloc((size_t)n * n * sizeof(double));   /* outside timed region */
    if (!C) { fprintf(stderr, "malloc failed\n"); return 2; }

    const char *kinds[2] = {"warmup", "measured"};
    int counts[2] = {warmup, reps};
    for (int kk = 0; kk < 2; kk++) {
        for (int r = 0; r < counts[kk]; r++) {
            int64_t t0 = now_ns();
            for (int b = 0; b < batch; b++) {
                matmul(A, B, C, n);
                __asm__ volatile("" ::: "memory");  /* each batch iteration must execute */
            }
            int64_t t1 = now_ns();
            printf("%s,%d,%lld,%d,%.17g\n", kinds[kk], r, (long long)(t1 - t0), batch, checksum(C, (size_t)n));
            fflush(stdout);
        }
    }
    if (out) save(out, C, (size_t)n);
    printf("#peak_rss_bytes,%lld\n", peak_rss_bytes());
    free(A); free(B); free(C);
    return 0;
}
