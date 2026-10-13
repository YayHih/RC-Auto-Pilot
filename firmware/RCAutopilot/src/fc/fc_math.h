/* fc_math.h -- deterministic single-precision math.
 *
 * Everything is built from IEEE-754 float + - * / (correctly rounded on every platform) and an
 * integer square root, never from the C library's sinf/atan2f/sqrtf, whose last bits differ between
 * the PC and newlib on the Cortex-M0+.  Compile with FMA contraction off (the M0+ has no FPU, the PC
 * build uses -ffp-contract=off) and the flight controller produces bit-identical outputs on both.
 */
#ifndef FC_MATH_H
#define FC_MATH_H
#include <stdint.h>
#include <string.h>

#define F_PI      3.14159265f
#define F_2PI     6.28318531f
#define F_PI_2    1.57079633f
#define F_D2R     0.0174532925f
#define F_G       9.80665f

static inline float f_clamp(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float f_abs(float x) { return x < 0.0f ? -x : x; }
static inline float f_min(float a, float b) { return a < b ? a : b; }
static inline float f_max(float a, float b) { return a > b ? a : b; }
static inline float f_sign(float x) { return x < 0.0f ? -1.0f : 1.0f; }

/* correctly rounded sqrt, bit-by-bit (fdlibm e_sqrtf method); 0 for x <= 0 or subnormal */
static inline float f_sqrt(float x)
{
    if (!(x > 0.0f)) return 0.0f;
    uint32_t ix;
    memcpy(&ix, &x, 4);
    int32_t m = (int32_t)(ix >> 23);
    if (m == 0) return 0.0f;
    if (m == 255) return x;
    m -= 127;
    ix = (ix & 0x007fffffu) | 0x00800000u;
    if (m & 1) ix += ix;
    m = (m - (m & 1)) / 2;
    ix += ix;
    uint32_t q = 0, s = 0, r = 0x01000000u;
    while (r) {
        uint32_t t = s + r;
        if (t <= ix) { s = t + r; ix -= t; q += r; }
        ix += ix;
        r >>= 1;
    }
    if (ix != 0) q += (q & 1u);
    ix = (q >> 1) + 0x3f000000u + ((uint32_t)m << 23);
    float z;
    memcpy(&z, &ix, 4);
    return z;
}

/* reduce to [-pi, pi] */
static inline float f_wrap(float a)
{
    float k = a * 0.159154943f;
    int32_t n = (int32_t)(k + (k >= 0.0f ? 0.5f : -0.5f));
    float fn = (float)n;
    return (a - fn * 6.28125f) - fn * 0.00193530717f;   /* 2*pi split for accuracy */
}

static inline float f_sin(float x)
{
    x = f_wrap(x);
    if (x > F_PI_2) x = F_PI - x;
    else if (x < -F_PI_2) x = -F_PI - x;
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.166666667f + x2 * (0.00833333333f + x2 * (-0.000198412698f + x2 * 2.75573192e-6f))));
}

static inline float f_cos(float x) { return f_sin(x + F_PI_2); }
static inline float f_tan(float x) { return f_sin(x) / f_cos(x); }

/* atan on [-1, 1], |err| < 2e-5 rad */
static inline float f_atan_unit(float z)
{
    float z2 = z * z;
    return z * (0.99997726f + z2 * (-0.33262347f + z2 * (0.19354346f + z2 * (-0.11643287f + z2 * (0.05265332f + z2 * -0.01172120f)))));
}

static inline float f_atan2(float y, float x)
{
    float ax = f_abs(x), ay = f_abs(y);
    if (ax == 0.0f && ay == 0.0f) return 0.0f;
    float a = (ay <= ax) ? f_atan_unit(ay / ax) : F_PI_2 - f_atan_unit(ax / ay);
    if (x < 0.0f) a = F_PI - a;
    return y < 0.0f ? -a : a;
}

static inline float f_atan(float z) { return f_atan2(z, 1.0f); }
static inline float f_asin(float s) { s = f_clamp(s, -1.0f, 1.0f); return f_atan2(s, f_sqrt(1.0f - s * s)); }
static inline float f_acos(float c) { c = f_clamp(c, -1.0f, 1.0f); return f_atan2(f_sqrt(1.0f - c * c), c); }

/* natural log for x > 0 (|err| ~1e-6) */
static inline float f_log(float x)
{
    if (!(x > 0.0f)) return -88.0f;
    uint32_t ix;
    memcpy(&ix, &x, 4);
    int32_t e = (int32_t)((ix >> 23) & 0xffu) - 127;
    ix = (ix & 0x007fffffu) | 0x3f800000u;     /* mantissa in [1, 2) */
    float m;
    memcpy(&m, &ix, 4);
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }
    float s = (m - 1.0f) / (m + 1.0f), s2 = s * s;
    float ln_m = 2.0f * s * (1.0f + s2 * (0.333333333f + s2 * (0.2f + s2 * (0.142857143f + s2 * 0.111111111f))));
    return (float)e * 0.693147181f + ln_m;
}

/* exp for |x| < ~10 by squaring a short series */
static inline float f_exp(float x)
{
    x = f_clamp(x, -20.0f, 20.0f);
    float y = x * 0.0625f;
    float r = 1.0f + y * (1.0f + y * (0.5f + y * (0.166666667f + y * (0.0416666667f + y * 0.00833333333f))));
    r *= r; r *= r; r *= r; r *= r;
    return r;
}

/* ---------------------------------------------------------------- small vector / quaternion kit */
typedef struct { float x, y, z; } v3;

static inline v3 v3_make(float x, float y, float z) { v3 r = {x, y, z}; return r; }
static inline v3 v3_add(v3 a, v3 b) { return v3_make(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline v3 v3_sub(v3 a, v3 b) { return v3_make(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline v3 v3_scale(v3 a, float k) { return v3_make(a.x * k, a.y * k, a.z * k); }
static inline float v3_dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline v3 v3_cross(v3 a, v3 b) { return v3_make(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline float v3_norm(v3 a) { return f_sqrt(v3_dot(a, a)); }

typedef struct { float w, x, y, z; } quat;

static inline quat q_mul(quat a, quat b)
{
    quat r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return r;
}

static inline quat q_conj(quat a) { quat r = {a.w, -a.x, -a.y, -a.z}; return r; }

static inline quat q_norm(quat a)
{
    float n = f_sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z);
    float k = n > 0.0f ? 1.0f / n : 1.0f;
    quat r = {a.w * k, a.x * k, a.y * k, a.z * k};
    return r;
}

/* 3x3 row-major body->NED */
typedef struct { float m[3][3]; } m33;

static inline m33 q_to_R(quat q)
{
    m33 R;
    float w = q.w, x = q.x, y = q.y, z = q.z;
    R.m[0][0] = 1 - 2 * (y * y + z * z); R.m[0][1] = 2 * (x * y - w * z);     R.m[0][2] = 2 * (x * z + w * y);
    R.m[1][0] = 2 * (x * y + w * z);     R.m[1][1] = 1 - 2 * (x * x + z * z); R.m[1][2] = 2 * (y * z - w * x);
    R.m[2][0] = 2 * (x * z - w * y);     R.m[2][1] = 2 * (y * z + w * x);     R.m[2][2] = 1 - 2 * (x * x + y * y);
    return R;
}

static inline v3 R_mul(const m33 *R, v3 v)
{
    return v3_make(R->m[0][0] * v.x + R->m[0][1] * v.y + R->m[0][2] * v.z,
                   R->m[1][0] * v.x + R->m[1][1] * v.y + R->m[1][2] * v.z,
                   R->m[2][0] * v.x + R->m[2][1] * v.y + R->m[2][2] * v.z);
}

static inline v3 RT_mul(const m33 *R, v3 v)
{
    return v3_make(R->m[0][0] * v.x + R->m[1][0] * v.y + R->m[2][0] * v.z,
                   R->m[0][1] * v.x + R->m[1][1] * v.y + R->m[2][1] * v.z,
                   R->m[0][2] * v.x + R->m[1][2] * v.y + R->m[2][2] * v.z);
}

static inline quat R_to_q(const m33 *Rm)
{
    const float (*R)[3] = Rm->m;
    float tr = R[0][0] + R[1][1] + R[2][2];
    quat q;
    if (tr > 0.0f) {
        float s = 2.0f * f_sqrt(tr + 1.0f);
        q.w = 0.25f * s; q.x = (R[2][1] - R[1][2]) / s; q.y = (R[0][2] - R[2][0]) / s; q.z = (R[1][0] - R[0][1]) / s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        float s = 2.0f * f_sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]);
        q.w = (R[2][1] - R[1][2]) / s; q.x = 0.25f * s; q.y = (R[0][1] + R[1][0]) / s; q.z = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        float s = 2.0f * f_sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]);
        q.w = (R[0][2] - R[2][0]) / s; q.x = (R[0][1] + R[1][0]) / s; q.y = 0.25f * s; q.z = (R[1][2] + R[2][1]) / s;
    } else {
        float s = 2.0f * f_sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]);
        q.w = (R[1][0] - R[0][1]) / s; q.x = (R[0][2] + R[2][0]) / s; q.y = (R[1][2] + R[2][1]) / s; q.z = 0.25f * s;
    }
    return q_norm(q);
}

/* ZYX Euler (yaw, pitch, roll) -> body->NED */
static inline m33 euler_to_R(float yaw, float pitch, float roll)
{
    float cy = f_cos(yaw), sy = f_sin(yaw), cp = f_cos(pitch), sp = f_sin(pitch), cr = f_cos(roll), sr = f_sin(roll);
    m33 R;
    R.m[0][0] = cy * cp; R.m[0][1] = cy * sp * sr - sy * cr; R.m[0][2] = cy * sp * cr + sy * sr;
    R.m[1][0] = sy * cp; R.m[1][1] = sy * sp * sr + cy * cr; R.m[1][2] = sy * sp * cr - cy * sr;
    R.m[2][0] = -sp;     R.m[2][1] = cp * sr;                R.m[2][2] = cp * cr;
    return R;
}

static inline void R_to_euler(const m33 *R, float *yaw, float *pitch, float *roll)
{
    *yaw = f_atan2(R->m[1][0], R->m[0][0]);
    *pitch = -f_asin(R->m[2][0]);
    *roll = f_atan2(R->m[2][1], R->m[2][2]);
}

/* attitude with body x along unit n and the belly facing horizontal heading hdg */
static inline m33 R_from_nose(v3 n, float hdg)
{
    v3 h = v3_make(f_cos(hdg), f_sin(hdg), 0.0f);
    v3 z = v3_sub(h, v3_scale(n, v3_dot(h, n)));
    z = v3_scale(z, 1.0f / f_max(v3_norm(z), 1e-6f));
    v3 y = v3_cross(z, n);
    m33 R;
    R.m[0][0] = n.x; R.m[0][1] = y.x; R.m[0][2] = z.x;
    R.m[1][0] = n.y; R.m[1][1] = y.y; R.m[1][2] = z.y;
    R.m[2][0] = n.z; R.m[2][1] = y.z; R.m[2][2] = z.z;
    return R;
}

#endif
