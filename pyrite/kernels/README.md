# Kernel backends

The transformer executor uses the Python reference path for normalization, RoPE, attention, routing, sampling, and tensor types without a native kernel. Optional portable scalar C kernels accelerate GGML Q4_K and Q6_K row-wise matrix-vector products and bounded row dequantization.

`native.c` has a plain C ABI and no Python, NumPy, BLAS, or Python development-header dependency. On Linux and macOS, `native.py` compiles it on first use with `cc`, `gcc`, or `clang`, caches the shared library in the system temporary directory, and calls it through `ctypes`. The quantized GGUF range is borrowed through CPython's buffer protocol; weights are not copied or expanded for matvec. If no compiler is available (or the platform is unsupported), the executor falls back to the existing Python reference decoder.

The current C implementation is deliberately scalar and portable. It is correctness-oriented, not a SIMD/throughput claim. T-SAR remains a separate reference ternary kernel; Pyrite does not implement that paper's proposed hardware changes.
