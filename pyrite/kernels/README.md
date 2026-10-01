# Kernel backends

This directory is reserved for optimized execution backends.

The first target is a portable CPU reference path. A future backend can implement a T-SAR-inspired ternary matrix kernel using the host CPU's SIMD capabilities. T-SAR is a hardware/software co-design paper; this repository does not claim to implement its proposed hardware modifications.
