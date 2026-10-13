/* estimator.cpp -- attitude (gyro + compass + GPS-velocity-aided tilt) and three 3-state Kalman
 * filters (position, velocity, accelerometer bias per NED axis) fed by GPS and baro.
 *
 * Tilt: an accelerometer cannot tell gravity from acceleration, and thrust, turns and pull-ups are
 * long sustained accelerations.  So the main tilt reference is GPS velocity: every fix, the velocity
 * change GPS saw over the last 0.5 s (allowing for its 100 ms latency) is compared with the integral
 * of the accelerometer, rotated by the attitude the filter had at the time, over exactly the same
 * window.  A mismatch can only come from a tilt error.  The plain "gravity is down" correction is
 * kept with a tight gate for the ground and for GPS dropouts. */
#include "fc_internal.h"

static const float KP_ACC = 2.0f, KI = 0.03f, KP_MAG = 0.8f;

void est_reset(Est *e)
{
    memset(e, 0, sizeof(*e));
    e->q.w = 1.0f;
    for (int i = 0; i < 3; i++) {
        e->P[i][0][0] = 4.0f; e->P[i][1][1] = 1.0f; e->P[i][2][2] = 0.05f;
    }
    e->acc = v3_make(0.0f, 0.0f, -F_G);
}

void est_boot_sample(Est *e, const v3 *acc, const v3 *mag, const float *baro_alt)
{
    if (acc) { e->acc_sum = v3_add(e->acc_sum, *acc); e->n_acc++; }
    if (mag) { e->mag_sum = v3_add(e->mag_sum, *mag); e->n_mag++; }
    if (baro_alt) { e->baro_sum += *baro_alt; e->n_baro++; }
}

/* level and north-align from the averaged boot samples; returns 1 when done */
int est_boot_finish(Est *e, float decl)
{
    if (e->n_acc < 50 || e->n_mag < 20 || e->n_baro < 10) return 0;
    v3 a = v3_scale(e->acc_sum, 1.0f / (float)e->n_acc);
    v3 m = v3_scale(e->mag_sum, 1.0f / (float)e->n_mag);
    v3 down = v3_scale(a, -1.0f / v3_norm(a));            /* at rest the specific force points up */
    v3 east = v3_cross(down, m);
    east = v3_scale(east, 1.0f / v3_norm(east));
    v3 north = v3_cross(east, down);
    m33 Rm;                                               /* rows: NED axes in body coords */
    Rm.m[0][0] = north.x; Rm.m[0][1] = north.y; Rm.m[0][2] = north.z;
    Rm.m[1][0] = east.x;  Rm.m[1][1] = east.y;  Rm.m[1][2] = east.z;
    Rm.m[2][0] = down.x;  Rm.m[2][1] = down.y;  Rm.m[2][2] = down.z;
    float c = f_cos(decl), s = f_sin(decl);               /* magnetic north sits at true azimuth decl */
    m33 R;
    for (int j = 0; j < 3; j++) {
        R.m[0][j] = c * Rm.m[0][j] - s * Rm.m[1][j];
        R.m[1][j] = s * Rm.m[0][j] + c * Rm.m[1][j];
        R.m[2][j] = Rm.m[2][j];
    }
    e->q = R_to_q(&R);
    v3 mn = R_mul(&R, m);
    e->m_ref = v3_scale(mn, 1.0f / v3_norm(mn));
    e->alt_baro0 = e->baro_sum / (float)e->n_baro;
    e->ready = 1;
    return 1;
}

static void kf_predict(float x[3], float P[3][3], float u, float dt)
{
    float d = dt, h = 0.5f * dt * dt;
    float x0 = x[0] + d * x[1] - h * x[2] + h * u;
    float x1 = x[1] - d * x[2] + d * u;
    x[0] = x0; x[1] = x1;
    /* P = F P F^T + Q for F = [[1, d, -h], [0, 1, -d], [0, 0, 1]], written out (13 multiplies, not 54) */
    float a = P[0][0], b = P[0][1], c = P[0][2], e = P[1][1], f = P[1][2], g = P[2][2];
    float r0x = a + d * b - h * c, r0y = b + d * e - h * f, r0z = c + d * f - h * g;
    float r1y = e - d * f, r1z = f - d * g;
    float p00 = r0x + d * r0y - h * r0z, p01 = r0y - d * r0z, p11 = r1y - d * r1z;
    float qv = 0.35f * dt, qb = 0.01f * dt;
    P[0][0] = p00; P[0][1] = p01; P[1][0] = p01; P[0][2] = r0z; P[2][0] = r0z;
    P[1][1] = p11 + qv * qv; P[1][2] = r1z; P[2][1] = r1z; P[2][2] = g + qb * qb;
}

/* N and E: same F, Q and measurement noise, so one covariance serves both */
static void kf_predict2(float xn[3], float xe[3], float P[3][3], float un, float ue, float dt)
{
    float d = dt, h = 0.5f * dt * dt;
    float n0 = xn[0] + d * xn[1] - h * xn[2] + h * un, n1 = xn[1] - d * xn[2] + d * un;
    float e0 = xe[0] + d * xe[1] - h * xe[2] + h * ue, e1 = xe[1] - d * xe[2] + d * ue;
    xn[0] = n0; xn[1] = n1; xe[0] = e0; xe[1] = e1;
    float x2[3] = {0.0f, 0.0f, 0.0f};
    float u = 0.0f;
    kf_predict(x2, P, u, dt);                              /* covariance part only */
}

/* measurement of state 0 (and optionally state 1) */
static void kf_update(float x[3], float P[3][3], int nmeas, float z0, float r0, float z1, float r1)
{
    if (nmeas == 1) {
        float S = P[0][0] + r0;
        float K[3] = {P[0][0] / S, P[1][0] / S, P[2][0] / S};
        float y = z0 - x[0];
        float P0[3] = {P[0][0], P[0][1], P[0][2]};
        for (int i = 0; i < 3; i++) {
            x[i] += K[i] * y;
            for (int j = 0; j < 3; j++) P[i][j] -= K[i] * P0[j];
        }
        return;
    }
    float s00 = P[0][0] + r0, s01 = P[0][1], s10 = P[1][0], s11 = P[1][1] + r1;
    float det = s00 * s11 - s01 * s10;
    float i00 = s11 / det, i01 = -s01 / det, i10 = -s10 / det, i11 = s00 / det;
    float K[3][2];
    for (int i = 0; i < 3; i++) {
        K[i][0] = P[i][0] * i00 + P[i][1] * i10;
        K[i][1] = P[i][0] * i01 + P[i][1] * i11;
    }
    float y0 = z0 - x[0], y1 = z1 - x[1];
    float Pr[2][3];
    for (int j = 0; j < 3; j++) { Pr[0][j] = P[0][j]; Pr[1][j] = P[1][j]; }
    for (int i = 0; i < 3; i++) {
        x[i] += K[i][0] * y0 + K[i][1] * y1;
        for (int j = 0; j < 3; j++) P[i][j] -= K[i][0] * Pr[0][j] + K[i][1] * Pr[1][j];
    }
}

/* horizontal GPS update: N and E share P, so compute the gain once and apply it to both states */
static void kf_update2(float xn[3], float xe[3], float P[3][3], float pn, float pe, float vn, float ve)
{
    const float r0 = 1.8f * 1.8f, r1 = 0.25f * 0.25f;
    float s00 = P[0][0] + r0, s01 = P[0][1], s10 = P[1][0], s11 = P[1][1] + r1;
    float det = s00 * s11 - s01 * s10;
    float i00 = s11 / det, i01 = -s01 / det, i10 = -s10 / det, i11 = s00 / det;
    float K[3][2];
    for (int i = 0; i < 3; i++) {
        K[i][0] = P[i][0] * i00 + P[i][1] * i10;
        K[i][1] = P[i][0] * i01 + P[i][1] * i11;
    }
    float yn0 = pn - xn[0], yn1 = vn - xn[1], ye0 = pe - xe[0], ye1 = ve - xe[1];
    float Pr[2][3];
    for (int j = 0; j < 3; j++) { Pr[0][j] = P[0][j]; Pr[1][j] = P[1][j]; }
    for (int i = 0; i < 3; i++) {
        xn[i] += K[i][0] * yn0 + K[i][1] * yn1;
        xe[i] += K[i][0] * ye0 + K[i][1] * ye1;
        for (int j = 0; j < 3; j++) P[i][j] -= K[i][0] * Pr[0][j] + K[i][1] * Pr[1][j];
    }
}

void est_imu(Est *e, v3 gyro, v3 acc, const v3 *mag, float dt, float thr, v3 motor_uT)
{
    e->thr = thr;
    m33 R = q_to_R(e->q);
    v3 up_b = v3_make(-R.m[2][0], -R.m[2][1], -R.m[2][2]);
    v3 m_up = acc;
    float fn = v3_norm(m_up);
    float width = e->gps_ok ? 0.06f : 0.25f;                  /* GPS aiding carries tilt in flight */
    float gate = f_max(0.0f, 1.0f - f_abs(fn / F_G - 1.0f) / width);
    if (e->gps_ok && v3_norm(gyro) > 0.3f) gate = 0.0f;     /* turning: centripetal, not gravity */
    v3 err = v3_make(0.0f, 0.0f, 0.0f);
    if (fn > 1e-3f && gate > 0.0f) err = v3_scale(v3_cross(v3_scale(m_up, 1.0f / fn), up_b), gate);
    if (mag) {
        v3 m = v3_sub(*mag, v3_scale(motor_uT, thr * thr));
        float mn = v3_norm(m);
        if (mn > 1e-3f) {
            v3 m_pred = RT_mul(&R, e->m_ref);
            v3 em = v3_cross(v3_scale(m, 1.0f / mn), m_pred);
            err = v3_add(err, v3_scale(up_b, KP_MAG / KP_ACC * v3_dot(em, up_b)));
        }
    }
    if (gate > 0.5f) {
        e->eint = v3_add(e->eint, v3_scale(err, dt));
        e->eint.x = f_clamp(e->eint.x, -0.5f, 0.5f);
        e->eint.y = f_clamp(e->eint.y, -0.5f, 0.5f);
        e->eint.z = f_clamp(e->eint.z, -0.5f, 0.5f);
    }
    /* GPS-velocity tilt correction, consumed at 1 /s */
    v3 tilt_b = RT_mul(&R, e->gps_tilt);
    err = v3_add(err, v3_scale(tilt_b, 1.0f / KP_ACC));
    if (e->gps_ok) {
        e->eint = v3_add(e->eint, v3_scale(tilt_b, 0.3f * dt));
        e->eint.x = f_clamp(e->eint.x, -0.5f, 0.5f);
        e->eint.y = f_clamp(e->eint.y, -0.5f, 0.5f);
    }
    e->gps_tilt = v3_scale(e->gps_tilt, 1.0f - dt);
    v3 w = v3_add(v3_add(gyro, v3_scale(err, KP_ACC)), v3_scale(e->eint, KI));
    quat wq = {0.0f, w.x, w.y, w.z};
    quat dq = q_mul(e->q, wq);
    quat q = {e->q.w + 0.5f * dq.w * dt, e->q.x + 0.5f * dq.x * dt, e->q.y + 0.5f * dq.y * dt, e->q.z + 0.5f * dq.z * dt};
    e->q = q_norm(q);
    e->w = v3_add(gyro, v3_scale(e->eint, KI));
    e->acc = acc;

    R = q_to_R(e->q);
    v3 rf = R_mul(&R, acc);
    e->hist_rf[e->n_tick & 63u] = rf;
    e->n_tick++;
    /* window for the GPS tilt check: ticks [n - lat - 50, n - lat), lat = GPS latency + processing delay.
     * Keeping a running sum (add the sample entering, drop the one leaving) costs 6 additions a tick. */
    {
        const uint32_t lat = 11;                           /* 10 ticks receiver latency + typically 1 tick deferral */
        if (e->n_tick >= lat + 1) e->win_sum = v3_add(e->win_sum, e->hist_rf[(e->n_tick - lat - 1) & 63u]);
        if (e->n_tick >= lat + 51) e->win_sum = v3_sub(e->win_sum, e->hist_rf[(e->n_tick - lat - 51) & 63u]);
    }
    v3 a_n = rf;
    a_n.z += F_G;
    e->a_ned_f = v3_add(e->a_ned_f, v3_scale(v3_sub(a_n, e->a_ned_f), 0.2f));
    kf_predict2(e->x[0], e->x[1], e->P[0], a_n.x, a_n.y, dt);     /* N and E share one covariance */
    memcpy(e->P[1], e->P[0], sizeof(e->P[0]));
    kf_predict(e->x[2], e->P[2], a_n.z, dt);
}

/* matched-window tilt measurement, called on each GPS velocity (10 Hz, 100 ms latency) */
static void gps_tilt_update(Est *e, v3 v, uint32_t L)
{
    for (int i = 7; i > 0; i--) e->vhist[i] = e->vhist[i - 1];
    e->vhist[0] = v;
    if (e->n_v < 8) e->n_v++;
    if (e->n_v < 6 || e->n_tick < 64) return;
    const uint32_t W = 50;                                 /* window, ticks */
    v3 F = v3_scale(e->win_sum, DT_TICK);                  /* maintained per tick in est_imu */
    (void)L;
    v3 dv_imu = v3_make(F.x, F.y, F.z + F_G * (float)W * DT_TICK);
    v3 dv_gps = v3_sub(e->vhist[0], e->vhist[5]);
    v3 err = v3_sub(dv_gps, dv_imu);
    float f2 = v3_dot(F, F);
    if (f2 < 1.0f) return;
    v3 dth = v3_scale(v3_cross(F, err), 1.0f / f2);
    dth.z = 0.0f;                                          /* tilt only: heading has its own sources */
    float n = v3_norm(dth);
    if (n > 0.2f) dth = v3_scale(dth, 0.2f / n);
    e->gps_tilt = dth;
}

void est_gps(Est *e, v3 p, v3 v, uint8_t defer)
{
    const float lat = 0.10f + (float)defer * DT_TICK;      /* receiver latency + our own deferral */
    gps_tilt_update(e, v, 10u + defer);
    if (e->have_prev) {
        v3 a_raw = v3_scale(v3_sub(v, e->v_gps_prev), 1.0f / 0.1f);
        e->a_gps = v3_add(e->a_gps, v3_scale(v3_sub(a_raw, e->a_gps), 0.25f));
    }
    e->v_gps_prev = v;
    e->have_prev = 1;
    p = v3_add(p, v3_scale(v, lat));                 /* latency compensation */
    v = v3_add(v, v3_scale(e->a_ned_f, lat));
    kf_update2(e->x[0], e->x[1], e->P[0], p.x, p.y, v.x, v.y);   /* N and E with one gain */
    memcpy(e->P[1], e->P[0], sizeof(e->P[0]));
    kf_update(e->x[2], e->P[2], 2, p.z, 3.0f * 3.0f, v.z, 0.4f * 0.4f);
    e->gps_ok = 1;
}

void est_baro(Est *e, float alt)
{
    kf_update(e->x[2], e->P[2], 1, -(alt - e->alt_baro0), 0.6f * 0.6f, 0.0f, 0.0f);
}

/* rotate the attitude (and the magnetic reference with it) about the NED down axis: the compass
 * heading error learned by the wind filter is fed back here so yaw itself becomes right */
void est_rotate_yaw(Est *e, float d)
{
    float h = 0.5f * d;
    quat qz = {f_cos(h), 0.0f, 0.0f, f_sin(h)};
    e->q = q_norm(q_mul(qz, e->q));
    float c = f_cos(d), s = f_sin(d);
    v3 m = e->m_ref;
    e->m_ref = v3_make(c * m.x - s * m.y, s * m.x + c * m.y, m.z);
}
