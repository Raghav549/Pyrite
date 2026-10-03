"""Optional portable C kernels for GGML matrices.

Pyrite compiles a tiny dependency-free shared library to the system temporary
cache on first use when a C compiler is present.  The library has a plain C
ABI (it does not require Python headers); compiler-less installs transparently
use the bounded Python reference implementation.  The checkpoint buffer is
borrowed through CPython's buffer protocol, so the native path does not copy or
expand quantized weights.

The kernels cover the plain (F32/F16/BF16/F64/I8-I64) and the most common
quantized layouts, including Q8_0 - the quantization used by the published
Qwen3 Q8_0 checkpoints.  ``NATIVE_TYPES`` mirrors the ``PYRITE_TYPES`` table in
``native.c`` and ``tests/test_native_kernels.py`` asserts the two agree, and
that every native decoder matches the Python reference decoder bit for bit.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from array import array
from collections.abc import Sequence
from pathlib import Path

#: GGML type ids with a native kernel.  Must match PYRITE_TYPES in native.c.
NATIVE_TYPES = frozenset({
    0,   # F32
    1,   # F16
    2,   # Q4_0
    3,   # Q4_1
    6,   # Q5_0
    7,   # Q5_1
    8,   # Q8_0
    9,   # Q8_1
    10,  # Q2_K
    12,  # Q4_K
    13,  # Q5_K
    14,  # Q6_K
    20,  # IQ4_NL
    24,  # I8
    25,  # I16
    26,  # I32
    27,  # I64
    28,  # F64
    30,  # BF16
})
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_ATTEMPTED = False


class _PyBuffer(ctypes.Structure):
    """CPython's stable Py_buffer layout, used only for a read-only simple view."""

    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("length", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_void_p),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("suboffsets", ctypes.c_void_p),
        ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = (ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int)
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = (ctypes.POINTER(_PyBuffer),)
_release_buffer.restype = None


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY, _LIBRARY_ATTEMPTED
    if _LIBRARY_ATTEMPTED:
        return _LIBRARY
    _LIBRARY_ATTEMPTED = True
    if sys.platform not in {"linux", "darwin"}:
        return None

    source = Path(__file__).with_name("native.c")
    if not source.is_file():
        return None
    try:
        source_bytes = source.read_bytes()
    except OSError:
        return None
    digest = hashlib.sha256(
        source_bytes + sys.platform.encode() + platform.machine().encode()
    ).hexdigest()[:20]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = Path(tempfile.gettempdir()) / f"pyrite-qkernels-{digest}{suffix}"
    try:
        if not output.is_file():
            compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
            if compiler is None:
                return None
            if sys.platform == "darwin":
                command = [compiler, "-O3", "-std=c99", "-fPIC", "-dynamiclib", str(source), "-lm", "-o", str(output)]
            else:
                temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
                command = [compiler, "-O3", "-std=c99", "-fPIC", "-shared", str(source), "-lm", "-o", str(temporary)]
                try:
                    # The compiler is selected from trusted PATH locations;
                    # every source and flag here is fixed by this module.
                    subprocess.run(command, check=True, capture_output=True, timeout=120)  # noqa: S603
                    os.replace(temporary, output)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                command = []
            if command:
                subprocess.run(command, check=True, capture_output=True, timeout=120)  # noqa: S603
        lib = ctypes.CDLL(str(output))
        pointer = ctypes.POINTER(ctypes.c_double)
        lib.pyrite_matvec.argtypes = (
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
        )
        lib.pyrite_matvec.restype = ctypes.c_int
        lib.pyrite_dequantize_rows.argtypes = (
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
        )
        lib.pyrite_dequantize_rows.restype = ctypes.c_int
        lib.pyrite_native_types.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)
        lib.pyrite_native_types.restype = ctypes.c_int
        count = lib.pyrite_native_types(None, 0)
        if count > 0:
            buffer = (ctypes.c_int * count)()
            written = lib.pyrite_native_types(buffer, count)
            reported = frozenset(buffer[i] for i in range(max(0, written)))
            # The C table is the source of truth; a mismatch means the two
            # halves of the kernel drifted apart, which would silently send
            # unsupported layouts down the native path.
            if reported != NATIVE_TYPES:
                return None
        _LIBRARY = lib
    except (OSError, subprocess.SubprocessError, AttributeError):
        _LIBRARY = None
    return _LIBRARY


def native_available() -> bool:
    return _load_library() is not None


def supports_native(ggml_type: int) -> bool:
    return ggml_type in NATIVE_TYPES and native_available()


def native_type_ids() -> tuple[int, ...]:
    """GGML type ids the loaded native library really implements."""
    lib = _load_library()
    if lib is None:
        return ()
    count = lib.pyrite_native_types(None, 0)
    if count <= 0:
        return ()
    buffer = (ctypes.c_int * count)()
    written = lib.pyrite_native_types(buffer, count)
    return tuple(buffer[i] for i in range(max(0, written)))


def sorted_native_types() -> list[str]:
    """Display names of the natively accelerated GGML types (empty if none)."""
    from ..ggml_types import type_name

    return [type_name(type_id) for type_id in sorted(native_type_ids())]


def _with_buffer(payload, callback):
    view = _PyBuffer()
    _get_buffer(payload, ctypes.byref(view), 0)  # PyBUF_SIMPLE: contiguous bytes
    try:
        return callback(view.buf, view.length)
    finally:
        _release_buffer(ctypes.byref(view))


def _kernel_error(code: int, operation: str) -> None:
    if code != 0:
        raise ValueError(f"native {operation} rejected the matrix buffer/shape (error {code})")


def matvec(
    ggml_type: int,
    payload: bytes | bytearray | memoryview,
    vector: Sequence[float],
    rows: int,
    cols: int,
) -> list[float]:
    """Fused quantized matrix-vector product without a dequantized matrix."""
    lib = _load_library()
    if ggml_type not in NATIVE_TYPES or lib is None:
        raise ValueError(f"no native kernel for GGML type {ggml_type}")
    if len(vector) != cols or rows <= 0 or cols <= 0:
        raise ValueError("matrix/vector shape mismatch")
    x = array("d", vector)
    result = array("d", [0.0]) * rows
    x_view = (ctypes.c_double * len(x)).from_buffer(x)
    out_view = (ctypes.c_double * len(result)).from_buffer(result)
    code = _with_buffer(
        payload,
        lambda address, length: lib.pyrite_matvec(
            ggml_type,
            address,
            length,
            x_view,
            out_view,
            rows,
            cols,
        ),
    )
    _kernel_error(code, "matvec")
    return result.tolist()


def dequantize_rows(
    ggml_type: int,
    payload: bytes | bytearray | memoryview,
    rows: int,
    cols: int,
) -> list[float]:
    """Dequantize only a bounded row range (used for embedding lookups)."""
    lib = _load_library()
    if ggml_type not in NATIVE_TYPES or lib is None:
        raise ValueError(f"no native kernel for GGML type {ggml_type}")
    if rows <= 0 or cols <= 0 or rows > sys.maxsize // cols:
        raise ValueError("invalid or overflowing matrix shape")
    result = array("d", [0.0]) * (rows * cols)
    out_view = (ctypes.c_double * len(result)).from_buffer(result)
    code = _with_buffer(
        payload,
        lambda address, length: lib.pyrite_dequantize_rows(
            ggml_type,
            address,
            length,
            out_view,
            rows,
            cols,
        ),
    )
    _kernel_error(code, "dequantize")
    return result.tolist()
