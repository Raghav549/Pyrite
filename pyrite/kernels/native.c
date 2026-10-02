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

static int matrix_geometry(int type_id, size_t rows, size_t cols, size_t *row_bytes,
                           size_t *expected_bytes) {
    size_t block_bytes;
    if (type_id == 12) {
        block_bytes = Q4K_BYTES;
    } else if (type_id == 14) {
        block_bytes = Q6K_BYTES;
    } else {
        return -1;
    }
    if (rows == 0 || cols == 0 || cols % QK_K != 0 ||
        cols / QK_K > SIZE_MAX / block_bytes) {
        return -2;
    }
    *row_bytes = (cols / QK_K) * block_bytes;
    if (*row_bytes > SIZE_MAX / rows) {
        return -2;
    }
    *expected_bytes = *row_bytes * rows;
    return 0;
}

int pyrite_matvec(int type_id, const uint8_t *data, size_t data_len,
                  const double *x, double *out, size_t rows, size_t cols) {
    size_t row_bytes = 0;
    size_t expected_bytes = 0;
    int geometry = matrix_geometry(type_id, rows, cols, &row_bytes, &expected_bytes);
    if (geometry != 0 || data == NULL || x == NULL || out == NULL || data_len != expected_bytes) {
        return geometry != 0 ? geometry : -3;
    }
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t *row_data = data + row * row_bytes;
        double sum = 0.0;
        if (type_id == 12) {
            for (size_t col = 0; col < cols; col += QK_K) {
                sum += dot_q4_k(row_data + (col / QK_K) * Q4K_BYTES, x + col);
            }
        } else {
            for (size_t col = 0; col < cols; col += QK_K) {
                sum += dot_q6_k(row_data + (col / QK_K) * Q6K_BYTES, x + col);
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
    int geometry = matrix_geometry(type_id, rows, cols, &row_bytes, &expected_bytes);
    if (geometry != 0 || data == NULL || out == NULL || data_len != expected_bytes ||
        cols > SIZE_MAX / rows) {
        return geometry != 0 ? geometry : -3;
    }
    for (size_t row = 0; row < rows; ++row) {
        double *row_out = out + row * cols;
        const uint8_t *row_data = data + row * row_bytes;
        for (size_t col = 0; col < cols; col += QK_K) {
            if (type_id == 12) {
                decode_q4_k(row_data + (col / QK_K) * Q4K_BYTES, row_out + col);
            } else {
                decode_q6_k(row_data + (col / QK_K) * Q6K_BYTES, row_out + col);
            }
        }
    }
    return 0;
}
