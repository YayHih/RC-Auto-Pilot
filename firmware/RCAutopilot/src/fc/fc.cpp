/* fc.cpp -- public API of the flight controller.
 *
 *   fc_default_config(&cfg)    fill a config with the nominal aeroplane / field
 *   fc_init(&cfg)              reset everything (call once at boot)
 *   fc_tick(&in, &out)         one 10 ms control step: raw sensor registers in, PCA9685 counts out
 *
 * Nothing here knows whether it runs on the Heltec, on a PC under the simulator, or inside the ARM
 * emulator: time comes in as in->t_us, sensors as raw register values, and the only output that
 * matters (what the servos see) is out->pca_off[].
 */
#include "fc_internal.h"
#include "fc.h"

static FC g_fc;

void fc_default_config(fc_config_t *c)
{
    memset(c, 0, sizeof(*c));
    c->pca_osc_hz = 25000000.0f;
    c->pca_prescale = 121;                  /* 25 MHz / (4096 * 122) = 50.03 Hz */
    for (int i = 0; i < 8; i++) { c->ch_center_us[i] = 1500; c->ch_throw_us[i] = 500; c->ch_reverse[i] = 1; }
    c->mag_decl_deg = -13.0f;
    c->mag_motor_uT[0] = 0.6f; c->mag_motor_uT[1] = 1.0f; c->mag_motor_uT[2] = -1.6f;
    c->auth_q0[0] = 4.4f;  c->auth_q0[1] = 1.04f; c->auth_q0[2] = 0.48f;
    c->auth_w0[0] = 26.0f; c->auth_w0[1] = 29.0f; c->auth_w0[2] = 14.0f;
    c->auth_lockin_cal[0] = 0.73f; c->auth_lockin_cal[1] = 0.80f; c->auth_lockin_cal[2] = 1.10f;
    c->auth_lockin_cal_w[0] = 1.02f; c->auth_lockin_cal_w[1] = 1.40f; c->auth_lockin_cal_w[2] = 1.32f;
    c->thrust_max_n = 22.9f;
    c->weight_n = 8.04f;
    c->thr_hover = 0.55f;
    c->rwy_len_ahead_m = 135.0f;
    c->rwy_len_behind_m = 75.0f;
    c->rwy_hdg_deg = -1.0f;
    c->cruise_alt_m = 28.0f;
    c->cruise_v = 13.0f;
    c->pattern_alt_m = 12.5f;            /* tuned: tools/tune.py over randomised aeroplanes */
    c->v_approach = 12.3f;
    c->glide_deg = 8.4f;
    c->fence_m = 15.0f;
    c->hold_r_m = 37.5f;
    c->flare_h_m = 1.5f;
    c->flare_tau_s = 1.2f;
    c->bw_fwd[0] = 20.8f; c->bw_fwd[1] = 18.2f; c->bw_fwd[2] = 18.2f;
    c->bw_hover[0] = 15.4f; c->bw_hover[1] = 30.7f; c->bw_hover[2] = 30.7f;
    c->att_kp = 5.2f;
    c->adapt_enable = 1;
    c->brake_enable = 1;
    c->fly_side = 1;
}

void fc_init(const fc_config_t *c)
{
    memset(&g_fc, 0, sizeof(g_fc));
    g_fc.cfg = *c;
    est_reset(&g_fc.est);
    wind_reset(&g_fc.wind);
    adapt_reset(&g_fc.adapt);
    ap_reset(&g_fc.ap, c);
    g_fc.pca_k = c->pca_osc_hz / (1000000.0f * (float)(c->pca_prescale + 1));
}

static int airborne(const FC *fc)
{
    uint8_t s = fc->ap.state;
    float alt = -fc->est.x[2][0];
    if (s == ST_INIT || s == ST_GROUND || s == ST_ROLLOUT) return 0;
    if (s == ST_TAKEOFF) return fc->ap.phase == 1 && alt > 1.0f;
    if (s == ST_MANUAL || s == ST_FLARE) return alt > 2.0f;
    return 1;
}

static uint16_t pca_count(const fc_config_t *c, float k, int ch, float cmd)
{
    float us = (float)c->ch_center_us[ch] + (float)c->ch_reverse[ch] * (float)c->ch_throw_us[ch] * f_clamp(cmd, -1.0f, 1.0f);
    float n = us * k + 0.5f;
    return (uint16_t)f_clamp(n, 0.0f, 4095.0f);
}

void fc_tick(const fc_in_t *in, fc_out_t *out)
{
    FC *fc = &g_fc;
    const fc_config_t *c = &fc->cfg;
    const float dt = DT_TICK;
    Est *e = &fc->est;

    /* ---- raw registers -> physical units */
    v3 acc = v3_make(0.0f, 0.0f, 0.0f), gyro = acc, mag = acc;
    float baro_alt = 0.0f;
    int has_imu = in->flags & FC_IN_IMU, has_mag = in->flags & FC_IN_MAG;
    int has_baro = in->flags & FC_IN_BARO, has_gps = (in->flags & FC_IN_GPS) && in->fix_type >= 3 && in->num_sv >= 5;
    if (has_imu) {
        const float ka = F_G / 4096.0f, kg = F_D2R / 32.8f;
        acc = v3_make((float)in->acc[0] * ka, (float)in->acc[1] * ka, (float)in->acc[2] * ka);
        gyro = v3_make((float)in->gyro[0] * kg, (float)in->gyro[1] * kg, (float)in->gyro[2] * kg);
    }
    if (has_mag) {
        const float km = 1.0f / 30.0f;          /* 3000 LSB/G, 1 G = 100 uT */
        mag = v3_make((float)in->mag[0] * km - c->mag_hard_iron_uT[0], (float)in->mag[1] * km - c->mag_hard_iron_uT[1],
                      (float)in->mag[2] * km - c->mag_hard_iron_uT[2]);
    }
    if (has_baro) {
        float p = (float)in->baro_q8 * (1.0f / 256.0f);
        baro_alt = 44330.0f * (1.0f - f_exp(0.190295f * f_log(p / 101325.0f)));
    }
    v3 gp = acc, gv = acc;
    if (has_gps) {
        if (!e->have_gps_home) {
            e->lat0 = in->lat_e7; e->lon0 = in->lon_e7; e->hmsl0 = in->hmsl_mm;
            e->m_per_e7_n = 0.0111194926f;                                     /* R_earth * pi/180 * 1e-7 */
            e->m_per_e7_e = 0.0111194926f * f_cos((float)in->lat_e7 * 1e-7f * F_D2R);
            e->have_gps_home = 1;
        }
        gp = v3_make((float)(in->lat_e7 - e->lat0) * e->m_per_e7_n, (float)(in->lon_e7 - e->lon0) * e->m_per_e7_e,
                     -(float)(in->hmsl_mm - e->hmsl0) * 0.001f);
        gv = v3_make((float)in->vel_mm_s[0] * 0.001f, (float)in->vel_mm_s[1] * 0.001f, (float)in->vel_mm_s[2] * 0.001f);
    }

    /* ---- estimation */
    if (!e->ready) {
        est_boot_sample(e, has_imu ? &acc : 0, has_mag ? &mag : 0, has_baro ? &baro_alt : 0);
        fc->boot_t += dt;
        if (fc->boot_t > 1.5f && est_boot_finish(e, c->mag_decl_deg * F_D2R)) fc->ap.state = ST_GROUND;
    } else {
        if (has_imu) est_imu(e, gyro, acc, has_mag ? &mag : 0, dt, fc->ap.out[3], v3_make(c->mag_motor_uT[0], c->mag_motor_uT[1], c->mag_motor_uT[2]));
        if (has_baro) est_baro(e, baro_alt);
    }
    /* GPS work is spread over light ticks (no compass, no baro): fix -> estimator -> wind filter */
    if (has_gps && e->ready) { fc->gps_pp = gp; fc->gps_pv = gv; fc->gps_stage = 1; fc->gps_age = 0; }
    else if (fc->gps_stage) fc->gps_age++;
    int light = !has_mag && !has_baro;
    int do_est = fc->gps_stage == 1 && (light || fc->gps_age >= 3);
    int do_wind = fc->gps_stage == 2 && (light || fc->gps_age >= 6);
    if (do_est) {
        est_gps(e, fc->gps_pp, fc->gps_pv, fc->gps_age);
        fc->gps_stage = 2;
    }
    v3 vel = v3_make(e->x[0][1], e->x[1][1], e->x[2][1]);
    fc->Vh = f_sqrt(vel.x * vel.x + vel.y * vel.y);
    m33 R = q_to_R(e->q);
    int fwd = airborne(fc) && fc->ap.state != ST_LOOP && fc->ap.state != ST_TO_HOVER && fc->ap.state != ST_HOVER
              && fc->ap.state != ST_FROM_HOVER && fc->ap.state != ST_TAKEOFF;
    if (fwd && has_imu) wind_predict(&fc->wind, acc.x, -R.m[2][0], dt);
    else if (!fwd) {   /* not in steady forward flight: airspeed = ground speed through the wind estimate */
        float gx = vel.x - fc->wind.x[0], gy = vel.y - fc->wind.x[1];
        fc->wind.x[2] = f_max(f_sqrt(gx * gx + gy * gy), 3.0f);
    }
    if (do_wind) {
        fc->gps_stage = 0;
        v3 gv = fc->gps_pv;
        float nh = f_sqrt(R.m[0][0] * R.m[0][0] + R.m[1][0] * R.m[1][0]);
        int usable = fwd && fc->Vh > 4.0f && nh > 0.6f && f_abs(R.m[2][1]) < 0.85f;
        /* angle of attack ~ load factor / dynamic pressure, scaled from the nominal 1 g value at 13 m/s */
        float Vw = f_max(fc->wind.x[2], 6.0f);
        float alpha = f_clamp(0.11f * (-e->acc.z / F_G) * (169.0f / (Vw * Vw)), -0.1f, 0.25f);
        wind_update(&fc->wind, gv, &R, alpha, 0.1f, usable);
        if (usable) fc->wind.h_ref += 0.05f * (f_max(-e->x[2][0], 2.0f) - fc->wind.h_ref);
        if (fc->wind.yaw_fix != 0.0f) {
            est_rotate_yaw(e, fc->wind.yaw_fix);
            /* a runway axis taken from the compass at line-up carries the same error */
            if (fc->ap.rwy_src == 0 && fc->ap.armed) {
                fc->ap.rwy_hdg = f_wrap(fc->ap.rwy_hdg + fc->wind.yaw_fix);
                fc->ap.land_dir = f_wrap(fc->ap.land_dir + fc->wind.yaw_fix);
            }
        }
    }
    /* the wind was measured up at circuit height; near the ground it is weaker (log boundary layer,
     * roughness z0 ~5 cm) - use the wind at our own height, or the airspeed reads high on short final */
    float wk = 1.0f;
    {
        float h = f_max(-e->x[2][0], 0.5f), hr = f_max(fc->wind.h_ref, 2.0f);
        if (h < hr) wk = f_clamp(f_log(h / 0.05f) / f_log(hr / 0.05f), 0.3f, 1.0f);
    }
    fc->wind_k = wk;
    v3 va = v3_make(vel.x - wk * fc->wind.x[0], vel.y - wk * fc->wind.x[1], vel.z);
    fc->V = v3_norm(va);
    fc->qbar = 0.5f * 1.225f * fc->V * fc->V;

    fc->gps_new = (uint8_t)(has_gps ? 1 : 0);
    fc->gps_v = gv;

    /* ---- guidance and control */
    ap_command(fc, in->cmd);
    ap_update(fc, in, dt);
    {
        /* control-authority identification: small dither in steady flight, lock-in measurement */
        float T = c->thrust_max_n * fc->ap.out[3] * fc->ap.out[3] * f_max(1.0f - fc->V / 28.0f, 0.0f);
        float tw = T / c->weight_n;
        uint8_t s = fc->ap.state;
        int steady = s == ST_CRUISE || s == ST_LOOP_PREP || s == ST_LAND || s == ST_HOVER;
        int en = c->adapt_enable && airborne(fc) && steady && -e->x[2][0] > 10.0f;
        float d[3];
        adapt_dither(&fc->adapt, c, fc->qbar, tw, en, d);
        for (int i = 0; i < 3; i++) fc->ap.out[i] = f_clamp(fc->ap.out[i] + d[i], -1.0f, 1.0f);
        float w[3] = {e->w.x, e->w.y, e->w.z};
        adapt_update(&fc->adapt, c, w, fc->ap.out, fc->qbar, tw, fc->V, en, light && !do_est && !do_wind);
    }

    /* ---- outputs: PCA9685 counts */
    const float *o = fc->ap.out;
    memset(out, 0, sizeof(*out));
    out->pca_off[0] = pca_count(c, fc->pca_k, 0, o[0]);
    out->pca_off[1] = pca_count(c, fc->pca_k, 1, o[1]);
    out->pca_off[2] = pca_count(c, fc->pca_k, 2, o[2]);
    out->pca_off[3] = pca_count(c, fc->pca_k, 3, 2.0f * o[3] - 1.0f);
    out->pca_off[4] = pca_count(c, fc->pca_k, 4, c->brake_enable ? 2.0f * o[4] - 1.0f : -1.0f);

    /* ---- telemetry */
    out->state = fc->ap.state;
    out->goal = fc->ap.goal;
    out->rc_mode = in->rc_mode;
    out->flags = (uint8_t)((fc->ap.armed ? 1 : 0) | (e->gps_ok ? 2 : 0) | (fc->wind.ok ? 4 : 0));
    for (int i = 0; i < 3; i++) { out->pos[i] = e->x[i][0]; out->vel[i] = e->x[i][1]; }
    out->q[0] = e->q.w; out->q[1] = e->q.x; out->q[2] = e->q.y; out->q[3] = e->q.z;
    out->rates[0] = e->w.x; out->rates[1] = e->w.y; out->rates[2] = e->w.z;
    out->wind[0] = fc->wind.x[0]; out->wind[1] = fc->wind.x[1];
    out->wind_sd = f_sqrt(0.5f * (fc->wind.P[0][0] + fc->wind.P[1][1]));
    out->airspeed = fc->V;
    for (int i = 0; i < 3; i++) { out->auth_q[i] = adapt_scale_q(&fc->adapt, i); out->auth_w[i] = adapt_scale_w(&fc->adapt, i); }
    for (int i = 0; i < 5; i++) out->cmd[i] = o[i];
    out->land_dir = fc->ap.land_dir;
    out->rwy_hdg = fc->ap.rwy_hdg;
    out->tick = ++fc->tick;
}

uint32_t fc_struct_sizes(void)
{
    return (uint32_t)sizeof(fc_in_t) | ((uint32_t)sizeof(fc_out_t) << 10) | ((uint32_t)sizeof(fc_config_t) << 20);
}
