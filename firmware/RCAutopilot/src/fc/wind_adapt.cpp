/* wind_adapt.cpp -- in-flight estimation of the wind and of the aeroplane's control authority.
 *
 * Wind (no airspeed sensor needed)
 *   GPS gives ground velocity, the attitude filter gives where the nose points.  With the air
 *   moving at w and the aeroplane flying at airspeed Va along its nose, tilted down by the angle of
 *   attack (estimated from the measured g-load):
 *       v_ground(horizontal) = Va * flow_direction_horizontal + w
 *   A 3-state Kalman filter (wN, wE, Va) solves this.  Between GPS fixes the airspeed is propagated
 *   with the accelerometer (dVa/dt = forward specific force - g sin(pitch)), so the filter knows when
 *   the aeroplane speeds up or slows down and only the wind is left to explain the rest.  Flying
 *   straight only shows the crosswind; every turn makes the along-track part observable too.
 *
 * Control authority (each aeroplane is different)
 *   Learning it from ordinary flight data is biased: turbulence kicks the aeroplane, the controller
 *   answers, and a regression reads that as "the surfaces do less than they do".  Instead every axis
 *   gets a small dither at its own frequency (roll 5 Hz, pitch 4 Hz, yaw 3 Hz) and a
 *   lock-in detector measures the angular-acceleration response at exactly that frequency over 2 s
 *   windows.  Disturbances are not phase-locked to the dither and average out.  The rate loop partly
 *   cancels the dither, so its amplitude is steered until the *net* surface motion is ~3.5 % of throw
 *   (well clear of servo backlash; < 1 deg of attitude wobble).  Each window gives
 *       A = rad/s^2 per unit command  at the current airspeed and throttle,
 *   corrected for servo lag and aerodynamic damping at the dither frequency, and a 2-parameter
 *   least-squares fit splits it into the two physical parts the controller schedules on:
 *       A = s_q * A_q0 * qbar  +  s_w * A_w0 * T/W        (A_q0, A_w0: nominal priors from ident.py)
 */
#include "fc_internal.h"

/* ---------------------------------------------------------------- wind */
/* states: wind N, wind E, airspeed, compass heading error d.  A compass in an aeroplane is several
 * degrees off (hard-iron residue, motor current, boot alignment); at 13 m/s 10 deg of heading error
 * is 2+ m/s of wind error.  The heading error is observable because its signature rotates with the
 * aeroplane in a turn while the wind does not.  It is fed back into the attitude filter (yaw_fix). */
void wind_reset(Wind *w)
{
    memset(w, 0, sizeof(*w));
    w->x[2] = 13.0f;
    w->h_ref = 20.0f;
    w->P[0][0] = 9.0f; w->P[1][1] = 9.0f; w->P[2][2] = 25.0f; w->P[3][3] = 0.04f;   /* ~11 deg */
}

/* every tick in forward flight: airspeed follows the measured along-nose acceleration */
void wind_predict(Wind *w, float fx, float sin_pitch, float dt)
{
    w->x[2] = f_clamp(w->x[2] + (fx - F_G * sin_pitch) * dt, 3.0f, 40.0f);
    w->P[2][2] = f_min(w->P[2][2] + 0.35f * 0.35f * dt, 25.0f);   /* accel bias, gusts, alpha changes */
}

void wind_update(Wind *w, v3 v, const m33 *R, float alpha, float dt, int usable)
{
    if (w->x[0] != w->x[0] || w->x[1] != w->x[1] || w->x[2] != w->x[2] || w->x[3] != w->x[3]) wind_reset(w);
    w->P[0][0] = f_min(w->P[0][0] + 0.1f * 0.1f * dt, 9.0f);
    w->P[1][1] = f_min(w->P[1][1] + 0.1f * 0.1f * dt, 9.0f);
    w->P[3][3] = f_min(w->P[3][3] + 0.004f * 0.004f * dt, 0.04f);
    w->yaw_fix = 0.0f;
    if (usable) {
        float hdg = f_atan2(R->m[1][0], R->m[0][0]);
        if (w->have_hdg) w->turned = f_min(w->turned + f_abs(f_wrap(hdg - w->last_hdg)), 100.0f);
        w->last_hdg = hdg;
        w->have_hdg = 1;
        float c = f_cos(w->x[3]), s = f_sin(w->x[3]);
        /* direction the air flows past: body (1, 0, alpha) -- the nose sits alpha above the flight
         * path, which in a banked turn points it inside the turn */
        float bx = R->m[0][0] + alpha * R->m[0][2], by = R->m[1][0] + alpha * R->m[1][2];
        float nx = bx * c - by * s;                         /* heading-corrected */
        float ny = bx * s + by * c;
        float Va = w->x[2];
        float H[2][4] = {{1.0f, 0.0f, nx, -Va * ny}, {0.0f, 1.0f, ny, Va * nx}};
        float PH[4][2];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                PH[i][j] = w->P[i][0] * H[j][0] + w->P[i][1] * H[j][1] + w->P[i][2] * H[j][2] + w->P[i][3] * H[j][3];
        const float r = 0.6f * 0.6f;                        /* GPS + sideslip + compass wobble */
        float S[2][2];
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                S[a][b] = H[a][0] * PH[0][b] + H[a][1] * PH[1][b] + H[a][2] * PH[2][b] + H[a][3] * PH[3][b] + (a == b ? r : 0.0f);
        float det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        if (!(det > 1e-9f)) return;
        float i00 = S[1][1] / det, i01 = -S[0][1] / det, i10 = -S[1][0] / det, i11 = S[0][0] / det;
        float K[4][2];
        for (int i = 0; i < 4; i++) {
            K[i][0] = PH[i][0] * i00 + PH[i][1] * i10;
            K[i][1] = PH[i][0] * i01 + PH[i][1] * i11;
        }
        float y0 = v.x - (w->x[0] + Va * nx);
        float y1 = v.y - (w->x[1] + Va * ny);
        /* 5-sigma gate on the innovation; if the measurements keep disagreeing it is the filter that is
         * wrong, not the GPS: re-seed the airspeed from ground speed minus wind and open P up again */
        float nis = y0 * (i00 * y0 + i01 * y1) + y1 * (i10 * y0 + i11 * y1);
        if (nis > 25.0f) {
            if (++w->n_reject >= 5) {
                float gx = v.x - w->x[0], gy = v.y - w->x[1];
                w->x[2] = f_clamp(f_sqrt(gx * gx + gy * gy), 3.0f, 40.0f);
                w->P[2][2] = 9.0f;
                w->P[0][0] = f_max(w->P[0][0], 4.0f);
                w->P[1][1] = f_max(w->P[1][1], 4.0f);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        if (i != j) w->P[i][j] = 0.0f;
                w->n_reject = 0;
            }
            return;
        }
        w->n_reject = 0;
        float HP[2][4];
        for (int j = 0; j < 4; j++) { HP[0][j] = PH[j][0]; HP[1][j] = PH[j][1]; }
        for (int i = 0; i < 4; i++) {
            w->x[i] += K[i][0] * y0 + K[i][1] * y1;
            for (int j = 0; j < 4; j++) w->P[i][j] -= K[i][0] * HP[0][j] + K[i][1] * HP[1][j];
        }
        w->x[2] = f_clamp(w->x[2], 3.0f, 40.0f);
        {   /* a wind faster than ~12 m/s is not one this aeroplane could be flying in */
            float wm = f_sqrt(w->x[0] * w->x[0] + w->x[1] * w->x[1]);
            if (wm > 12.0f) { w->x[0] *= 12.0f / wm; w->x[1] *= 12.0f / wm; }
        }
        w->x[3] = f_clamp(w->x[3], -0.6f, 0.6f);
        /* keep P symmetric and positive (single precision) */
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) { float m = 0.5f * (w->P[i][j] + w->P[j][i]); w->P[i][j] = m; w->P[j][i] = m; }
            if (!(w->P[i][i] > 1e-6f)) { wind_reset(w); return; }
        }
        /* hand the heading error to the attitude filter only once it is really observable: the
         * aeroplane has turned through >= 180 deg and the estimate is tight; small steps only */
        if (w->turned > F_PI && w->P[3][3] < 0.0015f) {
            float fix = f_clamp(0.5f * w->x[3], -0.01f, 0.01f);
            w->yaw_fix = fix;
            w->yaw_total += fix;
            w->x[3] -= fix;
        }
    }
    w->ok = (w->P[0][0] + w->P[1][1]) < 2.0f * 0.8f * 0.8f;
}

/* ---------------------------------------------------------------- control authority */
static const uint8_t AD_HZ[3] = {5, 4, 3};
/* sin(2 pi m / 100): the dither phases repeat exactly every 100 ticks, so a table (identical on every
 * platform) replaces 9 sin/cos evaluations per tick */
static const float SIN100[100] = {0.000000000f, 6.279051953e-02f, 1.253332336e-01f, 1.873813146e-01f, 2.486898872e-01f, 3.090169944e-01f, 3.681245527e-01f, 4.257792916e-01f, 4.817536741e-01f, 5.358267950e-01f, 5.877852523e-01f, 6.374239897e-01f, 6.845471059e-01f, 7.289686274e-01f, 7.705132428e-01f, 8.090169944e-01f, 8.443279255e-01f, 8.763066800e-01f, 9.048270525e-01f, 9.297764859e-01f, 9.510565163e-01f, 9.685831611e-01f, 9.822872507e-01f, 9.921147013e-01f, 9.980267284e-01f, 1.000000000f, 9.980267284e-01f, 9.921147013e-01f, 9.822872507e-01f, 9.685831611e-01f, 9.510565163e-01f, 9.297764859e-01f, 9.048270525e-01f, 8.763066800e-01f, 8.443279255e-01f, 8.090169944e-01f, 7.705132428e-01f, 7.289686274e-01f, 6.845471059e-01f, 6.374239897e-01f, 5.877852523e-01f, 5.358267950e-01f, 4.817536741e-01f, 4.257792916e-01f, 3.681245527e-01f, 3.090169944e-01f, 2.486898872e-01f, 1.873813146e-01f, 1.253332336e-01f, 6.279051953e-02f, 1.224646799e-16f, -6.279051953e-02f, -1.253332336e-01f, -1.873813146e-01f, -2.486898872e-01f, -3.090169944e-01f, -3.681245527e-01f, -4.257792916e-01f, -4.817536741e-01f, -5.358267950e-01f, -5.877852523e-01f, -6.374239897e-01f, -6.845471059e-01f, -7.289686274e-01f, -7.705132428e-01f, -8.090169944e-01f, -8.443279255e-01f, -8.763066800e-01f, -9.048270525e-01f, -9.297764859e-01f, -9.510565163e-01f, -9.685831611e-01f, -9.822872507e-01f, -9.921147013e-01f, -9.980267284e-01f, -1.000000000f, -9.980267284e-01f, -9.921147013e-01f, -9.822872507e-01f, -9.685831611e-01f, -9.510565163e-01f, -9.297764859e-01f, -9.048270525e-01f, -8.763066800e-01f, -8.443279255e-01f, -8.090169944e-01f, -7.705132428e-01f, -7.289686274e-01f, -6.845471059e-01f, -6.374239897e-01f, -5.877852523e-01f, -5.358267950e-01f, -4.817536741e-01f, -4.257792916e-01f, -3.681245527e-01f, -3.090169944e-01f, -2.486898872e-01f, -1.873813146e-01f, -1.253332336e-01f, -6.279051953e-02f};
static inline float tsin(uint16_t k, int i) { return SIN100[(AD_HZ[i] * k) % 100u]; }
static inline float tcos(uint16_t k, int i) { return SIN100[(AD_HZ[i] * k + 25u) % 100u]; }
static const float AD_UNET = 0.035f;
static const float AD_D0[3] = {-0.85f, -0.6f, -0.3f};     /* nominal damping, per (m/s), rad/s^2 per rad/s */
static const float SERVO_TAU = 0.035f;

void adapt_reset(Adapt *a)
{
    memset(a, 0, sizeof(*a));
    for (int i = 0; i < 3; i++) {
        a->s[i][0] = 1.0f; a->s[i][1] = 1.0f;
        a->amp[i] = 0.04f;
        a->P[i][0][0] = 0.25f; a->P[i][1][1] = 0.25f;
    }
}

/* the dither to add to this tick's commands */
void adapt_dither(const Adapt *a, const fc_config_t *c, float qbar, float tw, int enable, float d[3])
{
    (void)c; (void)qbar; (void)tw;
    for (int i = 0; i < 3; i++)
        d[i] = enable ? a->amp[i] * tsin(a->k, i) : 0.0f;
}

void adapt_update(Adapt *a, const fc_config_t *c, const float w[3], const float u[3], float qbar,
                  float tw, float V, int enable)
{
    if (!enable || !a->have_prev) {           /* restart the window */
        for (int i = 0; i < 3; i++) a->w_prev[i] = w[i];
        a->have_prev = 1;
        a->k = 0;
        a->valid = 1;
        for (int i = 0; i < 3; i++) { a->ys[i] = a->yc[i] = a->us[i] = a->uc[i] = a->rq[i] = a->rw[i] = 0.0f; }
        a->Vsum = 0.0f;
        if (!enable) a->have_prev = 0;
        return;
    }
    for (int i = 0; i < 3; i++) {
        float sn = tsin(a->k, i), cs = tcos(a->k, i);
        float wd = (w[i] - a->w_prev[i]) * (1.0f / DT_TICK);
        a->w_prev[i] = w[i];
        a->ys[i] += wd * sn; a->yc[i] += wd * cs;
        a->us[i] += u[i] * sn; a->uc[i] += u[i] * cs;
        a->rq[i] += c->auth_q0[i] * qbar;
        a->rw[i] += c->auth_w0[i] * tw;
        if (f_abs(u[i]) > 0.9f) a->valid = 0;              /* saturated: not a linear measurement */
    }
    a->Vsum += V;
    if (++a->k < AD_WIN) return;

    if (a->valid) {
        const float invN = 1.0f / (float)AD_WIN;
        float Vm = a->Vsum * invN;
        for (int i = 0; i < 3; i++) {
            float Y = f_sqrt(a->ys[i] * a->ys[i] + a->yc[i] * a->yc[i]);
            float U = f_sqrt(a->us[i] * a->us[i] + a->uc[i] * a->uc[i]);
            float Unet = U * 2.0f * invN;
            a->amp[i] = f_clamp(a->amp[i] * f_clamp(AD_UNET / f_max(Unet, 1e-3f), 0.5f, 2.0f), 0.02f, 0.12f);
            if (Unet < 0.6f * AD_UNET) continue;         /* too small yet: backlash would bias it */
            float Om = F_2PI * (float)AD_HZ[i];
            float servo = 1.0f / f_sqrt(1.0f + Om * SERVO_TAU * Om * SERVO_TAU);
            float D = AD_D0[i] * Vm;
            float phi[2] = {a->rq[i] * invN, a->rw[i] * invN};
            float cal = (c->auth_lockin_cal[i] * phi[0] + c->auth_lockin_cal_w[i] * phi[1]) / f_max(phi[0] + phi[1], 1e-6f);
            float A = Y / (U * servo) * f_sqrt(1.0f + (D / Om) * (D / Om)) / cal;
            /* only learn the part this flight condition actually exercises */
            if (phi[1] < 0.25f * phi[0]) phi[1] = 0.0f;
            else if (phi[0] < 0.25f * phi[1]) phi[0] = 0.0f;
            /* normalise by the expected measurement scatter (~15 %) */
            float sig = 0.15f * f_max(A, 8.0f);
            phi[0] /= sig; phi[1] /= sig;
            float y = A / sig;
            float (*P)[2] = a->P[i];
            float Pp0 = P[0][0] * phi[0] + P[0][1] * phi[1];
            float Pp1 = P[1][0] * phi[0] + P[1][1] * phi[1];
            const float lam = 0.97f;
            float den = lam + phi[0] * Pp0 + phi[1] * Pp1;
            float e = y - (a->s[i][0] * phi[0] + a->s[i][1] * phi[1]);
            float K0 = Pp0 / den, K1 = Pp1 / den;
            a->s[i][0] = f_clamp(a->s[i][0] + K0 * e, 0.3f, 3.0f);
            a->s[i][1] = f_clamp(a->s[i][1] + K1 * e, 0.3f, 3.0f);
            float p00 = (P[0][0] - K0 * Pp0) / (phi[0] != 0.0f ? lam : 1.0f);
            float p11 = (P[1][1] - K1 * Pp1) / (phi[1] != 0.0f ? lam : 1.0f);
            float p01 = (phi[0] != 0.0f && phi[1] != 0.0f) ? (P[0][1] - K0 * Pp1) / lam : 0.0f;
            P[0][0] = f_min(p00, 0.25f); P[0][1] = p01; P[1][0] = p01; P[1][1] = f_min(p11, 0.25f);
            a->windows[i]++;
        }
    }
    a->k = 0;
    a->valid = 1;
    a->Vsum = 0.0f;
    for (int i = 0; i < 3; i++) { a->ys[i] = a->yc[i] = a->us[i] = a->uc[i] = a->rq[i] = a->rw[i] = 0.0f; }
}

float adapt_scale_q(const Adapt *a, int i) { return a->s[i][0]; }
float adapt_scale_w(const Adapt *a, int i) { return a->s[i][1]; }
