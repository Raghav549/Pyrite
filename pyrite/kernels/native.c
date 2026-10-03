/*
 * Portable, header-only scalar CPU kernels for GGML Q4_K and Q6_K matrices.
 *
 * The routines mirror the public block layouts in ggml-common.h and the
 * dequantization order in ggml-quants.c, fused with a row-wise dot product.
 * Quantized weights are read directly from a borrowed buffer; no expanded
 * matrix is allocated.  This file deliberately has no Python, BLAS, NumPy, or
 * GPU dependency so Pyrite can build it as a small C shared library on demand.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define QK_K 256
#define Q4K_BYTES 144
#define Q6K_BYTES 210

static uint16_t read_le16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fU;
    uint32_t mant = (uint32_t)h & 0x03ffU;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffU;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fU) {
        bits = sign | 0x7f800000U | (mant << 13);
    } else {
        bits = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void get_scale_min_k4(int j, const uint8_t *scales, int *scale, int *minimum) {
    if (j < 4) {
        *scale = scales[j] & 63;
        *minimum = scales[j + 4] & 63;
    } else {
        *scale = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        *minimum = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

static double dot_q4_k(const uint8_t *block, const double *x) {
    const double d = (double)fp16_to_float(read_le16(block));
    const double dmin = (double)fp16_to_float(read_le16(block + 2));
    const uint8_t *scales = block + 4;
    const uint8_t *quants = block + 16;
    double sum = 0.0;
    int is = 0;
    int q = 0;

    for (int j = 0; j < QK_K; j += 64) {
        int sc1, m1, sc2, m2;
        get_scale_min_k4(is, scales, &sc1, &m1);
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const double d1 = d * (double)sc1;
        const double min1 = dmin * (double)m1;
        const double d2 = d * (double)sc2;
        const double min2 = dmin * (double)m2;
        for (int l = 0; l < 32; ++l) {
            const uint8_t packed = quants[q + l];
            const double w0 = d1 * (double)(packed & 0x0f) - min1;
            const double w1 = d2 * (double)(packed >> 4) - min2;
            sum += w0 * x[j + l];
            sum += w1 * x[j + l + 32];
        }
        q += 32;
        is += 2;
    }
    return sum;
}

static double dot_q6_k(const uint8_t *block, const double *x) {
    const uint8_t *ql = block;
    const uint8_t *qh = block + 128;
    const int8_t *scales = (const int8_t *)(block + 192);
    const double d = (double)fp16_to_float(read_le16(block + 208));
    double sum = 0.0;

    for (int half = 0; half < 2; ++half) {
        const int ql_off = half * 64;
        const int qh_off = half * 32;
        const int sc_off = half * 8;
        const int x_off = half * 128;
        for (int l = 0; l < 32; ++l) {
            const int sc_group = l / 16;
            const uint8_t low = ql[ql_off + l];
            const uint8_t low_next = ql[ql_off + l + 32];
            const uint8_t high = qh[qh_off + l];
            const int q1 = (int)(low & 0x0f) | (int)((high >> 0) & 0x03) << 4;
            const int q2 = (int)(low_next & 0x0f) | (int)((high >> 2) & 0x03) << 4;
            const int q3 = (int)(low >> 4) | (int)((high >> 4) & 0x03) << 4;
            const int q4 = (int)(low_next >> 4) | (int)((high >> 6) & 0x03) << 4;
            const int8_t s1 = scales[sc_off + sc_group + 0];
            const int8_t s2 = scales[sc_off + sc_group + 2];
            const int8_t s3 = scales[sc_off + sc_group + 4];
            const int8_t s4 = scales[sc_off + sc_group + 6];
            sum += (d * (double)s1 * (double)(q1 - 32)) * x[x_off + l];
            sum += (d * (double)s2 * (double)(q2 - 32)) * x[x_off + l + 32];
            sum += (d * (double)s3 * (double)(q3 - 32)) * x[x_off + l + 64];
            sum += (d * (double)s4 * (double)(q4 - 32)) * x[x_off + l + 96];
        }
    }
    return sum;
}

static void decode_q4_k(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const double dmin = (double)fp16_to_float(read_le16(block + 2));
    const uint8_t *scales = block + 4;
    const uint8_t *quants = block + 16;
    int is = 0;
    int q = 0;

    for (int j = 0; j < QK_K; j += 64) {
        int sc1, m1, sc2, m2;
        get_scale_min_k4(is, scales, &sc1, &m1);
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const double d1 = d * (double)sc1;
        const double min1 = dmin * (double)m1;
        const double d2 = d * (double)sc2;
        const double min2 = dmin * (double)m2;
        for (int l = 0; l < 32; ++l) {
            const uint8_t packed = quants[q + l];
            out[j + l] = d1 * (double)(packed & 0x0f) - min1;
            out[j + l + 32] = d2 * (double)(packed >> 4) - min2;
        }
        q += 32;
        is += 2;
    }
}

static void decode_q6_k(const uint8_t *block, double *out) {
    const uint8_t *ql = block;
    const uint8_t *qh = block + 128;
    const int8_t *scales = (const int8_t *)(block + 192);
    const double d = (double)fp16_to_float(read_le16(block + 208));

    for (int half = 0; half < 2; ++half) {
        const int ql_off = half * 64;
        const int qh_off = half * 32;
        const int sc_off = half * 8;
        const int out_off = half * 128;
        for (int l = 0; l < 32; ++l) {
            const int sc_group = l / 16;
            const uint8_t low = ql[ql_off + l];
            const uint8_t low_next = ql[ql_off + l + 32];
            const uint8_t high = qh[qh_off + l];
            const int q1 = (int)(low & 0x0f) | (int)((high >> 0) & 0x03) << 4;
            const int q2 = (int)(low_next & 0x0f) | (int)((high >> 2) & 0x03) << 4;
            const int q3 = (int)(low >> 4) | (int)((high >> 4) & 0x03) << 4;
            const int q4 = (int)(low_next >> 4) | (int)((high >> 6) & 0x03) << 4;
            out[out_off + l] = d * (double)scales[sc_off + sc_group + 0] * (double)(q1 - 32);
            out[out_off + l + 32] = d * (double)scales[sc_off + sc_group + 2] * (double)(q2 - 32);
            out[out_off + l + 64] = d * (double)scales[sc_off + sc_group + 4] * (double)(q3 - 32);
            out[out_off + l + 96] = d * (double)scales[sc_off + sc_group + 6] * (double)(q4 - 32);
        }
    }
}

/* ------------------------------------------------------------------ *
 * Additional decoders, mirroring pyrite/tensor_ops.py (which mirrors
 * ggml-quants.c).  Every routine here is checked against the Python
 * reference by tests/test_native_kernels.py, so a layout mistake shows
 * up as a test failure rather than as a silently wrong model.
 * ------------------------------------------------------------------ */

typedef void (*decode_block_fn)(const uint8_t *block, double *out);

#define GGML_HALF_BITS 16

static void decode_f32(const uint8_t *block, double *out) {
    uint32_t bits;
    memcpy(&bits, block, 4);
    float value;
    memcpy(&value, &bits, sizeof(value));
    out[0] = (double)value;
}

static void decode_f64(const uint8_t *block, double *out) {
    memcpy(out, block, 8);
}

static void decode_f16(const uint8_t *block, double *out) {
    out[0] = (double)fp16_to_float(read_le16(block));
}

static void decode_bf16(const uint8_t *block, double *out) {
    uint32_t bits = ((uint32_t)read_le16(block)) << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    out[0] = (double)value;
}

static void decode_i8(const uint8_t *block, double *out) { out[0] = (double)(int8_t)block[0]; }

static void decode_i16(const uint8_t *block, double *out) {
    out[0] = (double)(int16_t)read_le16(block);
}

static void decode_i32(const uint8_t *block, double *out) {
    uint32_t bits = (uint32_t)block[0] | ((uint32_t)block[1] << 8) |
                    ((uint32_t)block[2] << 16) | ((uint32_t)block[3] << 24);
    out[0] = (double)(int32_t)bits;
}

static void decode_i64(const uint8_t *block, double *out) {
    uint64_t bits = 0;
    for (int i = 7; i >= 0; --i) {
        bits = (bits << 8) | (uint64_t)block[i];
    }
    out[0] = (double)(int64_t)bits;
}

static void decode_q4_0(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const uint8_t *qs = block + 2;
    for (int i = 0; i < 16; ++i) {
        out[i] = d * (double)((int)(qs[i] & 0x0f) - 8);
        out[i + 16] = d * (double)((int)(qs[i] >> 4) - 8);
    }
}

static void decode_q4_1(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const double m = (double)fp16_to_float(read_le16(block + 2));
    const uint8_t *qs = block + 4;
    for (int i = 0; i < 16; ++i) {
        out[i] = d * (double)(qs[i] & 0x0f) + m;
        out[i + 16] = d * (double)(qs[i] >> 4) + m;
    }
}

static void decode_q5_0(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const uint32_t qh = (uint32_t)block[2] | ((uint32_t)block[3] << 8) |
                        ((uint32_t)block[4] << 16) | ((uint32_t)block[5] << 24);
    const uint8_t *qs = block + 6;
    for (int i = 0; i < 16; ++i) {
        const int low = (int)(qs[i] & 0x0f);
        const int high = (int)(qs[i] >> 4);
        /* ggml: xh_0 = ((qh >> i) << 4) & 0x10, xh_1 = (qh >> (i + 12)) & 0x10,
         * i.e. bit i and bit i+16 of the qh mask; both halves are symmetric
         * around -16 before scaling. */
        out[i] = d * ((double)(low | (int)(((qh >> i) & 1u) << 4)) - 16.0);
        out[i + 16] = d * ((double)(high | (int)(((qh >> (i + 16)) & 1u) << 4)) - 16.0);
    }
}

static void decode_q5_1(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const double m = (double)fp16_to_float(read_le16(block + 2));
    const uint32_t qh = (uint32_t)block[4] | ((uint32_t)block[5] << 8) |
                        ((uint32_t)block[6] << 16) | ((uint32_t)block[7] << 24);
    const uint8_t *qs = block + 8;
    for (int i = 0; i < 16; ++i) {
        const int low = (int)(qs[i] & 0x0f);
        const int high = (int)(qs[i] >> 4);
        out[i] = d * (double)(low | (int)(((qh >> i) & 1u) << 4)) + m;
        out[i + 16] = d * (double)(high | (int)(((qh >> (i + 16)) & 1u) << 4)) + m;
    }
}

static void decode_q8_0(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const int8_t *qs = (const int8_t *)(block + 2);
    for (int i = 0; i < 32; ++i) {
        out[i] = d * (double)qs[i];
    }
}

static void decode_q8_1(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const int8_t *qs = (const int8_t *)(block + 4);
    for (int i = 0; i < 32; ++i) {
        out[i] = d * (double)qs[i];
    }
}

/* kvalues_iq4nl from ggml-quants.c: the 4-bit non-linear code book. */
static const int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static void decode_iq4_nl(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const uint8_t *qs = block + 2;
    for (int i = 0; i < 16; ++i) {
        out[i] = d * (double)KVALUES_IQ4NL[qs[i] & 0x0f];
        out[i + 16] = d * (double)KVALUES_IQ4NL[qs[i] >> 4];
    }
}

static void decode_q2_k(const uint8_t *block, double *out) {
    const uint8_t *scales = block;
    const uint8_t *quants = block + 16;
    const double d = (double)fp16_to_float(read_le16(block + 80));
    const double dmin = (double)fp16_to_float(read_le16(block + 82));
    int is = 0;
    int q = 0;
    for (int outer = 0; outer < 256; outer += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            double dl = d * (double)(scales[is] & 0x0f);
            double ml = dmin * (double)(scales[is] >> 4);
            ++is;
            for (int l = 0; l < 16; ++l) {
                out[outer + shift * 16 + l] = dl * (double)((quants[q + l] >> shift) & 3) - ml;
            }
            dl = d * (double)(scales[is] & 0x0f);
            ml = dmin * (double)(scales[is] >> 4);
            ++is;
            for (int l = 0; l < 16; ++l) {
                out[outer + shift * 16 + 16 + l] =
                    dl * (double)((quants[q + 16 + l] >> shift) & 3) - ml;
            }
            shift += 2;
        }
        q += 32;
    }
}

static void decode_q5_k(const uint8_t *block, double *out) {
    const double d = (double)fp16_to_float(read_le16(block));
    const double dmin = (double)fp16_to_float(read_le16(block + 2));
    const uint8_t *scales = block + 4;
    const uint8_t *qh = block + 16;
    const uint8_t *ql = block + 48;
    int is = 0;
    int q = 0;
    unsigned int u1 = 1, u2 = 2;
    for (int outer = 0; outer < 256; outer += 64) {
        int sc1, m1, sc2, m2;
        get_scale_min_k4(is + 0, scales, &sc1, &m1);
        get_scale_min_k4(is + 1, scales, &sc2, &m2);
        const double d1 = d * (double)sc1;
        const double min1 = dmin * (double)m1;
        const double d2 = d * (double)sc2;
        const double min2 = dmin * (double)m2;
        for (int l = 0; l < 32; ++l) {
            out[outer + l] = d1 * (double)((ql[q + l] & 0x0f) + ((qh[l] & u1) ? 16 : 0)) - min1;
            out[outer + l + 32] =
                d2 * (double)((ql[q + l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - min2;
        }
        q += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/* --------------------------------------------------------------- *
 * Type table.  ids and geometry match ggml/include/ggml.h and
 * ggml/src/ggml-common.h; they are also asserted against
 * pyrite/ggml_types.py by tests/test_native_kernels.py.
 * --------------------------------------------------------------- */
typedef struct {
    int type_id;
    size_t block_size;
    size_t bytes_per_block;
    decode_block_fn decode;
} pyrite_type;

static const pyrite_type PYRITE_TYPES[] = {
    { 0,  1,   4, decode_f32    },
    { 1,  1,   2, decode_f16    },
    { 2,  32,  18, decode_q4_0  },
    { 3,  32,  20, decode_q4_1  },
    { 6,  32,  22, decode_q5_0  },
    { 7,  32,  24, decode_q5_1  },
    { 8,  32,  34, decode_q8_0  },
    { 9,  32,  36, decode_q8_1  },
    { 10, 256, 84, decode_q2_k  },
    { 12, 256, 144, decode_q4_k },
    { 13, 256, 176, decode_q5_k },
    { 14, 256, 210, decode_q6_k },
    { 20, 32,  18, decode_iq4_nl },
    { 24, 1,   1, decode_i8     },
    { 25, 1,   2, decode_i16    },
    { 26, 1,   4, decode_i32    },
    { 27, 1,   8, decode_i64    },
    { 28, 1,   8, decode_f64    },
    { 30, 1,   2, decode_bf16   },
};

static const pyrite_type *lookup_type(int type_id) {
    for (size_t i = 0; i < sizeof(PYRITE_TYPES) / sizeof(PYRITE_TYPES[0]); ++i) {
        if (PYRITE_TYPES[i].type_id == type_id) {
            return &PYRITE_TYPES[i];
        }
    }
    return NULL;
}

int pyrite_native_types(int *ids, size_t capacity) {
    const size_t count = sizeof(PYRITE_TYPES) / sizeof(PYRITE_TYPES[0]);
    if (ids == NULL) {
        return (int)count;
    }
    const size_t n = count < capacity ? count : capacity;
    for (size_t i = 0; i < n; ++i) {
        ids[i] = PYRITE_TYPES[i].type_id;
    }
    return (int)n;
}

static int matrix_geometry(int type_id, size_t rows, size_t cols, size_t *row_bytes,
                           size_t *expected_bytes, const pyrite_type **type_out) {
    const pyrite_type *type = lookup_type(type_id);
    if (type == NULL || type->decode == NULL) {
        return -1;
    }
    if (rows == 0 || cols == 0 || type->block_size == 0 ||
        cols % type->block_size != 0 ||
        cols / type->block_size > SIZE_MAX / type->bytes_per_block) {
        return -2;
    }
    *row_bytes = (cols / type->block_size) * type->bytes_per_block;
    if (*row_bytes > SIZE_MAX / rows) {
        return -2;
    }
    *expected_bytes = *row_bytes * rows;
    if (type_out != NULL) {
        *type_out = type;
    }
    return 0;
}

int pyrite_matvec(int type_id, const uint8_t *data, size_t data_len,
                  const double *x, double *out, size_t rows, size_t cols) {
    size_t row_bytes = 0;
    size_t expected_bytes = 0;
    const pyrite_type *type = NULL;
    int geometry = matrix_geometry(type_id, rows, cols, &row_bytes, &expected_bytes, &type);
    if (geometry != 0 || data == NULL || x == NULL || out == NULL || data_len != expected_bytes) {
        return geometry != 0 ? geometry : -3;
    }
    if (type->block_size == 1) {
        /* Unquantized rows: a single flat dot product is the fastest form. */
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t *row_data = data + row * row_bytes;
            double sum = 0.0;
            for (size_t col = 0; col < cols; ++col) {
                double value;
                type->decode(row_data + col * type->bytes_per_block, &value);
                sum += value * x[col];
            }
            out[row] = sum;
        }
        return 0;
    }
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t *row_data = data + row * row_bytes;
        double sum = 0.0;
        double scratch[QK_K];
        for (size_t col = 0; col < cols; col += type->block_size) {
            const uint8_t *block = row_data + (col / type->block_size) * type->bytes_per_block;
            type->decode(block, scratch);
            for (size_t l = 0; l < type->block_size; ++l) {
                sum += scratch[l] * x[col + l];
            }
        }
        out[row] = sum;
    }
    return 0;
}

int pyrite_dequantize_rows(int type_id, const uint8_t *data, size_t data_len,
                           double *out, size_t rows, size_t cols) {
    size_t row_bytes = 0;
    size_t expected_bytes = 0;
    const pyrite_type *type = NULL;
    int geometry = matrix_geometry(type_id, rows, cols, &row_bytes, &expected_bytes, &type);
    if (geometry != 0 || data == NULL || out == NULL || data_len != expected_bytes ||
        cols > SIZE_MAX / rows) {
        return geometry != 0 ? geometry : -3;
    }
    for (size_t row = 0; row < rows; ++row) {
        double *row_out = out + row * cols;
        const uint8_t *row_data = data + row * row_bytes;
        for (size_t col = 0; col < cols; col += type->block_size) {
            type->decode(row_data + (col / type->block_size) * type->bytes_per_block,
                         row_out + col);
        }
    }
    return 0;
}
