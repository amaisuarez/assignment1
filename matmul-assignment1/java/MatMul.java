import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Basic dense matrix multiplication, triple loop (i-j-k), Java.
 *
 * Storage: contiguous row-major double[] (element (i,j) at i*n + j), same as C.
 * Since Java 17 floating-point semantics are strict IEEE 754 (JEP 306), and the
 * JIT does not fuse a*b+c into FMA unless Math.fma is called explicitly.
 *
 * Output format (stdout), one line per repetition:
 *     kind,index,elapsed_ns,batch,checksum
 */
public final class MatMul {

    static double[] load(String path, int n) throws IOException {
        byte[] bytes = Files.readAllBytes(Path.of(path));
        if (bytes.length != 8L * n * n)
            throw new IOException(path + ": expected " + (long) n * n + " values");
        double[] m = new double[n * n];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(m);
        return m;
    }

    static void save(String path, double[] C) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * C.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.asDoubleBuffer().put(C);
        Files.write(Path.of(path), buf.array());
    }

    static void matmul(double[] A, double[] B, double[] C, int n) {
        for (int i = 0; i < n; i++) {
            int row = i * n;
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++)
                    s += A[row + k] * B[k * n + j];
                C[row + j] = s;
            }
        }
    }

    static double checksum(double[] C) {
        double s = 0.0;
        for (double x : C) s += x;
        return s;
    }

    /** Peak resident set size of this process (Linux VmHWM); -1 if unavailable. */
    static long peakRssBytes() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status")))
                if (line.startsWith("VmHWM:"))
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
        } catch (IOException | RuntimeException e) { /* not Linux */ }
        return -1;
    }

    public static void main(String[] args) throws IOException {
        int n = -1, warmup = 0, reps = 1, batch = 1;
        String pa = null, pb = null, out = null;
        for (int i = 0; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--n" -> n = Integer.parseInt(args[i + 1]);
                case "--a" -> pa = args[i + 1];
                case "--b" -> pb = args[i + 1];
                case "--warmup" -> warmup = Integer.parseInt(args[i + 1]);
                case "--reps" -> reps = Integer.parseInt(args[i + 1]);
                case "--batch" -> batch = Integer.parseInt(args[i + 1]);
                case "--out" -> out = args[i + 1];
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (n <= 0 || pa == null || pb == null) {
            System.err.println("usage: MatMul --n N --a A.bin --b B.bin [--warmup W] [--reps R] [--batch K] [--out C.bin]");
            System.exit(1);
        }
        if ((long) n * n > Integer.MAX_VALUE) throw new IllegalArgumentException("n too large for a flat double[]");

        double[] A = load(pa, n);
        double[] B = load(pb, n);
        double[] C = new double[n * n];          // outside timed region

        String[] kinds = {"warmup", "measured"};
        int[] counts = {warmup, reps};
        for (int kk = 0; kk < 2; kk++) {
            for (int r = 0; r < counts[kk]; r++) {
                long t0 = System.nanoTime();     // monotonic
                for (int b = 0; b < batch; b++) matmul(A, B, C, n);
                long t1 = System.nanoTime();
                System.out.println(kinds[kk] + "," + r + "," + (t1 - t0) + "," + batch + "," + checksum(C));
            }
        }
        if (out != null) save(out, C);
        Runtime rt = Runtime.getRuntime();
        System.out.println("#peak_rss_bytes," + peakRssBytes());
        System.out.println("#java_heap_used_bytes," + (rt.totalMemory() - rt.freeMemory()));
    }
}
