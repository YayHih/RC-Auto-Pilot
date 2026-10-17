/* fc_internal.h -- flight-controller modules and their state.  All state lives in one static
 * FC struct (no heap, no constructors), so the RAM use is fixed and visible in the map file. */
#ifndef FC_INTERNAL_H
#define FC_INTERNAL_H
#include "fc_types.h"
#include "fc_math.h"

#define DT_TICK 0.01f

/* ---------------------------------------------------------------- estimator */
typedef struct {
    quat q;
    v3 eint, w, acc, a_ned_f, a_gps, v_gps_prev, m_ref;
    float x[3][3];          /* per NED axis: pos, vel, accel bias */
    float P[3][3][3];
    float alt_baro0, thr;
    uint8_t gps_ok, have_prev, ready, have_gps_home;
    /* boot-time averaging */
    v3 acc_sum, mag_sum;
    float baro_sum;
    uint16_t n_acc, n_mag, n_baro;
    int32_t lat0, lon0, hmsl0;
    float m_per_e7_n, m_per_e7_e;
    /* GPS-velocity tilt aiding: per-tick R*f history and the last GPS velocities (matched windows) */
    v3 hist_rf[64];
    v3 vhist[8];
    uint32_t n_tick;
    uint8_t n_v;
    v3 gps_tilt;            /* tilt error (NED rotation vector) still to be corrected */
    v3 win_sum;             /* running sum of hist_rf over the tilt window */
} Est;

void est_reset(Est *e);
void est_boot_sample(Est *e, const v3 *acc, const v3 *mag, const float *baro_alt);
int  est_boot_finish(Est *e, float decl_rad);
void est_imu(Est *e, v3 gyro, v3 acc, const v3 *mag, float dt, float thr, v3 motor_uT);
void est_gps(Est *e, v3 p, v3 v, uint8_t defer_ticks);
void est_baro(Est *e, float alt);
void est_rotate_yaw(Est *e, float dpsi);

/* ---------------------------------------------------------------- wind estimator */
typedef struct {
    float x[4];             /* wind N, wind E, airspeed, compass heading error (rad) */
    float P[4][4];
    float yaw_fix;          /* heading correction to hand to the attitude filter     */
    float yaw_total;        /* total correction applied so far (telemetry)           */
    float turned;           /* heading change flown while usable (observability)    */
    float last_hdg;
    float h_ref;            /* height the wind was measured at (smoothed)            */
    uint8_t ok, have_hdg, n_reject;
} Wind;

void wind_reset(Wind *w);
void wind_predict(Wind *w, float fx, float sin_pitch, float dt);
void wind_update(Wind *w, v3 vel_ned, const m33 *R, float alpha, float dt, int usable);

/* ---------------------------------------------------------------- control-authority learning */
#define AD_WIN 200          /* 2 s lock-in window (whole periods of 3, 4 and 5 Hz) */
typedef struct {
    float s[3][2];          /* learned scale: [axis][airspeed part, prop-wash part]   */
    float P[3][2][2];
    float ys[3], yc[3], us[3], uc[3];    /* lock-in sums of dw/dt and command          */
    float rq[3], rw[3];                  /* regressor sums over the window             */
    float Vsum;
    float w_prev[3];
    uint16_t k;             /* tick within window */
    uint8_t valid, have_prev;
    float amp[3];           /* dither amplitude, steered so the net surface motion is ~3.5 % */
    uint8_t surprise[3];    /* consecutive windows that disagreed with the estimate (change detector) */
    /* finished window, evaluated one axis per tick so it never lands on one tick with GPS work */
    float wys[3], wyc[3], wus[3], wuc[3], wrq[3], wrw[3], wVm;
    uint8_t pend;
    uint16_t windows[3];    /* accepted windows per axis (telemetry) */
} Adapt;

void adapt_reset(Adapt *a);
void adapt_dither(const Adapt *a, const fc_config_t *c, float qbar, float tw, int enable, float d[3]);
void adapt_update(Adapt *a, const fc_config_t *c, const float w[3], const float u[3], float qbar,
                  float tw, float V, int enable, int light_tick);
float adapt_scale_q(const Adapt *a, int i);
float adapt_scale_w(const Adapt *a, int i);

/* ---------------------------------------------------------------- autopilot */
typedef struct {
    uint8_t state, goal, phase, final_ok, wind_side, ga_count;
    float t_state;
    float int_rate[3];
    float thr_h, hover_int, th_cmd, hdg_hold, alt_tgt, loop_theta0;
    float gam_prev, gam_unw, slip_f, glide_int, spd_int, touch_t, slow_t;
    float hover_tgt[2], nose_h_f[2];
    float home[2], rwy_hdg, land_dir;
    float rwy_c, rwy_s;     /* cos/sin of rwy_hdg, refreshed each tick (trig is ~3k cycles on the M0+) */
    float out[5];           /* da de dr thr brake */
    uint8_t armed, prev_rc, rwy_src, exit_ok;   /* exit_ok: hover has turned to a safe exit heading */     /* rwy_src: 0 line-up compass, 1 GPS roll track, 2 config */
    float trk_n, trk_e;
} AP;

typedef struct {
    fc_config_t cfg;
    Est est;
    Wind wind;
    Adapt adapt;
    AP ap;
    uint32_t tick;
    uint32_t t_us_prev;
    float V, Vh, qbar;      /* airspeed from wind estimate, ground speed, dynamic pressure */
    float yaw, pitch, roll; /* Euler angles of the attitude estimate, computed once per tick */
    float pca_k;            /* PCA9685 counts per microsecond */
    float wind_k;           /* wind at our height / wind where it was measured */
    v3 gps_v;               /* raw GPS velocity of this tick (gps_new) */
    uint8_t gps_new;
    /* GPS work is spread over the light (odd, no-compass) ticks: fix -> estimator -> wind filter */
    v3 gps_pp, gps_pv;
    uint8_t gps_stage, gps_age;
    float boot_t;
} FC;

void ap_reset(AP *ap, const fc_config_t *c);
void ap_command(FC *fc, uint8_t cmd);
void ap_update(FC *fc, const fc_in_t *in, float dt);

#endif
