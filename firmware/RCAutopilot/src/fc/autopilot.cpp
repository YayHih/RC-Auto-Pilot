/* autopilot.cpp -- button-driven state machine: TAKEOFF / CRUISE (orbit) / LOOP / HOVER / LAND,
 * plus MANUAL (RC direct or angle-assisted).  Inputs come only from the estimators.
 *
 * Landing:  LAND   pick the runway direction with the most headwind (wind estimate), fly to a holding
 *                  circle beside the extended centreline and spiral down to circuit height there
 *           FINAL  leave the circle tangentially onto the centreline; pitch holds approach airspeed,
 *                  throttle holds the glide path; go around if not stabilised by 8 m
 *           FLARE  idle, pitch controls sink rate (exponential flare), heading aligned with the runway
 *           ROLLOUT nose-wheel/rudder steers back to the centreline, brakes, stop on GPS speed
 *
 * Safety: the pilot stands on one side of the runway.  Everything except the take-off, the final
 * approach and a go-around (which are over the runway) is flown on the other side (cfg.fly_side),
 * at least cfg.fence_m from the centreline: the loiter orbit, the loop line, the hover and the landing
 * holding circle.  A soft fence turns the aeroplane away whenever guidance would take it closer.
 */
#include "fc_internal.h"

static const float RHO = 1.225f;
static const float ORBIT_S = 90.0f, ORBIT_R = 60.0f;
static const float LOOP_ALT = 50.0f, LOOP_S = 60.0f;
static const float AIM_IN = 25.0f;

/* ------------------------------------------------------------------ helpers */
static v3 est_pos(const FC *fc) { return v3_make(fc->est.x[0][0], fc->est.x[1][0], fc->est.x[2][0]); }
static v3 est_vel(const FC *fc) { return v3_make(fc->est.x[0][1], fc->est.x[1][1], fc->est.x[2][1]); }
static float est_alt(const FC *fc) { return -fc->est.x[2][0]; }

static void euler(const FC *fc, float *yaw, float *pitch, float *roll)
{
    *yaw = fc->yaw; *pitch = fc->pitch; *roll = fc->roll;     /* computed once per tick in ap_update */
}

static void go(AP *ap, uint8_t s)
{
    ap->state = s;
    ap->t_state = 0.0f;
    ap->int_rate[0] = ap->int_rate[1] = ap->int_rate[2] = 0.0f;
}

void ap_reset(AP *ap, const fc_config_t *c)
{
    memset(ap, 0, sizeof(*ap));
    ap->state = ST_INIT;
    ap->thr_h = c->thr_hover;
    ap->alt_tgt = c->cruise_alt_m;
    ap->prev_rc = FC_RC_AUTO;
}

/* control power (rad/s^2 per unit command) = learned airspeed part + learned prop-wash part */
static void authority(const FC *fc, float V, float thr, float out[3])
{
    const fc_config_t *c = &fc->cfg;
    float T = c->thrust_max_n * thr * thr * f_max(1.0f - V / 28.0f, 0.0f);
    float tw = T / c->weight_n;
    float qb = 0.5f * RHO * V * V;
    for (int i = 0; i < 3; i++)
        out[i] = adapt_scale_q(&fc->adapt, i) * c->auth_q0[i] * qb + adapt_scale_w(&fc->adapt, i) * c->auth_w0[i] * tw;
}

static v3 att_ctrl(const FC *fc, const m33 *R_des, v3 w_ff, float yaw_gain, int use_r_sp, float r_sp)
{
    quat qd = R_to_q(R_des);
    quat qe = q_mul(q_conj(fc->est.q), qd);
    if (qe.w < 0.0f) { qe.w = -qe.w; qe.x = -qe.x; qe.y = -qe.y; qe.z = -qe.z; }
    float kp = fc->cfg.att_kp;
    v3 w_sp = v3_make(f_clamp(kp * 2.0f * qe.x, -9.0f, 9.0f) + w_ff.x,
                      f_clamp(kp * 2.0f * qe.y, -9.0f, 9.0f) + w_ff.y,
                      f_clamp(yaw_gain * 2.0f * qe.z, -4.0f, 4.0f) + w_ff.z);
    if (use_r_sp) w_sp.z = r_sp;
    return w_sp;
}

static void rate_ctrl(FC *fc, v3 w_sp, float thr, float V, const float bw[3], float u[3])
{
    float auth[3];
    authority(fc, V, thr, auth);
    float sp[3] = {w_sp.x, w_sp.y, w_sp.z};
    float w[3] = {fc->est.w.x, fc->est.w.y, fc->est.w.z};
    for (int i = 0; i < 3; i++) {
        float Kr = bw[i] / f_max(auth[i], 12.0f);
        float err = sp[i] - w[i];
        /* integrator trims out steady torques (prop torque, slipstream swirl); light leak */
        fc->ap.int_rate[i] = f_clamp(fc->ap.int_rate[i] * (1.0f - 0.3f * DT_TICK) + err * DT_TICK * 4.0f * Kr, -0.4f, 0.4f);
        u[i] = f_clamp(Kr * err + fc->ap.int_rate[i], -1.0f, 1.0f);
    }
}

#define BW_FWD (fc->cfg.bw_fwd)
#define BW_HOVER (fc->cfg.bw_hover)

/* conventional attitude hold: tracks roll/pitch, yaw coordinated (or holds a heading if hold) */
static void fwd_att(FC *fc, float phi, float theta, float thr, float V, int hold, float hdg_hold, float turn_rate, float u[3])
{
    float yaw, pit, rol;
    euler(fc, &yaw, &pit, &rol);
    m33 Rd = euler_to_R(yaw, theta, phi);
    v3 w_ff = v3_make(0.0f, turn_rate * f_sin(phi), 0.0f);
    float r_sp = hold ? 3.0f * f_wrap(hdg_hold - yaw) : turn_rate * f_cos(phi);
    v3 w_sp = att_ctrl(fc, &Rd, w_ff, 6.0f, 1, r_sp);
    rate_ctrl(fc, w_sp, thr, V, BW_FWD, u);
    fc->ap.slip_f += 0.05f * (fc->est.acc.y - fc->ap.slip_f);
    u[2] = f_clamp(u[2] - 0.06f * fc->ap.slip_f, -1.0f, 1.0f);
}

static float course(const FC *fc)
{
    v3 v = est_vel(fc);
    if (f_sqrt(v.x * v.x + v.y * v.y) > 2.0f) return f_atan2(v.y, v.x);
    float y, p, r;
    euler(fc, &y, &p, &r);
    return y;
}

static void bank_from_heading(const FC *fc, float hdg, float V, float lim, float *phi, float *tr)
{
    float err = f_wrap(hdg - course(fc));
    *phi = f_clamp(1.4f * err, -lim, lim);
    *tr = F_G * f_tan(*phi) / f_max(V, 6.0f);
}

/* L1-style circle following; dir +1 clockwise (seen from above), -1 anticlockwise */
static float orbit_heading(v3 pos, float cn, float ce, float R, float dir)
{
    float dn = pos.x - cn, de = pos.y - ce;
    float dist = f_sqrt(dn * dn + de * de);
    float beta = f_atan2(de, dn);
    return f_wrap(beta + dir * F_PI_2 + dir * f_atan((dist - R) / 30.0f));
}

/* heading to track the line through (n0,e0) along hdg_line; xt > 0 = right of the line */
static float line_heading(v3 pos, float n0, float e0, float hdg_line, float L1, float *along, float *xt)
{
    float dn = pos.x - n0, de = pos.y - e0;
    float c = f_cos(hdg_line), s = f_sin(hdg_line);
    *along = dn * c + de * s;
    *xt = -dn * s + de * c;
    float d = f_clamp(-f_atan(*xt / L1), -1.0f, 1.0f);
    return f_wrap(hdg_line + d);
}

/* energy control: pitch for height rate, throttle for speed; both trims learn the aeroplane */
static void tecs(FC *fc, float alt_t, float V_t, float V, float *theta, float *thr)
{
    AP *ap = &fc->ap;
    float alt = est_alt(fc), vz = -est_vel(fc).z;
    float sink_max = (alt > fc->cfg.pattern_alt_m + 10.0f && alt_t < alt) ? 3.5f : 2.0f;
    float vz_t = f_clamp(0.5f * (alt_t - alt), -sink_max, 3.0f);
    float pitch_trim = 0.06f;
    *theta = f_clamp(pitch_trim + 0.08f * (vz_t - vz) + 0.03f * vz_t, -0.35f, 0.40f);
    ap->spd_int = f_clamp(ap->spd_int + 0.015f * (V_t - V) * DT_TICK, -0.25f, 0.30f);   /* learns cruise throttle */
    *thr = f_clamp(0.52f + ap->spd_int + 0.10f * (V_t - V) + 0.03f * (vz_t - vz), 0.05f, 1.0f);
}

static void rwy_point(const AP *ap, float s, float t, float *n, float *e)
{
    float c = ap->rwy_c, sn = ap->rwy_s;
    *n = ap->home[0] + s * c - t * sn;
    *e = ap->home[1] + s * sn + t * c;
}

/* signed distance from the runway centreline, positive on the side we are allowed to fly */
static float side_dist(const FC *fc, v3 p)
{
    const AP *ap = &fc->ap;
    float dn = p.x - ap->home[0], de = p.y - ap->home[1];
    float t = -dn * ap->rwy_s + de * ap->rwy_c;
    return (float)fc->cfg.fly_side * t;
}

/* soft fence: closer than `margin` to the pilot's side, turn the heading away from it */
static float fence(const FC *fc, v3 p, float hdg, float margin)
{
    float d = side_dist(fc, p);
    if (d >= margin) return hdg;
    float away = f_wrap(fc->ap.rwy_hdg + (float)fc->cfg.fly_side * F_PI_2);
    float w = f_clamp((margin - d) / 4.0f, 0.0f, 1.0f);
    return f_wrap(hdg + w * f_wrap(away - hdg));
}

/* bank for a heading, but a big turn near the runway always goes the way that first carries the
 * aeroplane towards the flying side, even if that is the long way round */
static void bank_side(const FC *fc, v3 p, float hdg, float V, float lim, float *phi, float *tr)
{
    float crs = course(fc);
    float err = f_wrap(hdg - crs);
    if (f_abs(err) > 1.4f && side_dist(fc, p) < fc->cfg.fence_m + 2.5f * V * V / (F_G * f_tan(lim) + 1e-3f)) {
        /* velocity change of a right turn (heading increasing) is along (-sin, cos) of the course */
        float fs = (float)fc->cfg.fly_side;
        float good = fs * (f_sin(crs) * fc->ap.rwy_s + f_cos(crs) * fc->ap.rwy_c);
        if (good > 0.0f && err < 0.0f) err += F_2PI;        /* turn right */
        else if (good < 0.0f && err > 0.0f) err -= F_2PI;   /* turn left */
    }
    *phi = f_clamp(1.4f * err, -lim, lim);
    *tr = F_G * f_tan(*phi) / f_max(V, 6.0f);
}

/* the transition out of the hover dives towards the belly heading: never towards the pilot.  Keep it
 * within ~70 deg of straight away from the runway (parallel to it at worst). */
static float safe_exit(const FC *fc, float h)
{
    float away = f_wrap(fc->ap.rwy_hdg + (float)fc->cfg.fly_side * F_PI_2);
    float d = f_wrap(h - away);
    const float lim = F_PI_2 - 0.35f;
    return f_abs(d) <= lim ? h : f_wrap(away + f_sign(d) * lim);
}

static float hold_s(const fc_config_t *c)
{
    return -c->pattern_alt_m / f_tan(c->glide_deg * F_D2R) - 15.0f;
}

/* landing frame: origin = aim point, s along the landing direction, t to its right */
static void land_origin(const FC *fc, float *n, float *e)
{
    const AP *ap = &fc->ap;
    int same = f_abs(f_wrap(ap->land_dir - ap->rwy_hdg)) < F_PI_2;
    float a = same ? -fc->cfg.rwy_len_behind_m + AIM_IN : fc->cfg.rwy_len_ahead_m - AIM_IN;
    rwy_point(ap, a, 0.0f, n, e);
}

static void land_frame(const FC *fc, v3 p, float *s, float *t)
{
    float on, oe;
    land_origin(fc, &on, &oe);
    float c = f_cos(fc->ap.land_dir), sn = f_sin(fc->ap.land_dir);
    *s = (p.x - on) * c + (p.y - oe) * sn;
    *t = -(p.x - on) * sn + (p.y - oe) * c;
}

static void choose_land_dir(FC *fc)
{
    AP *ap = &fc->ap;
    float wn = fc->wind.x[0], we = fc->wind.x[1];
    float c = f_cos(ap->rwy_hdg), s = f_sin(ap->rwy_hdg);
    float tail = wn * c + we * s;                          /* wind component along the runway heading */
    ap->land_dir = (fc->wind.ok && tail > 0.7f) ? f_wrap(ap->rwy_hdg + F_PI) : ap->rwy_hdg;
}

/* ------------------------------------------------------------------ buttons */
static void begin_takeoff(FC *fc)
{
    AP *ap = &fc->ap;
    float y, p, r;
    euler(fc, &y, &p, &r);
    v3 pos = est_pos(fc);
    ap->home[0] = pos.x; ap->home[1] = pos.y;
    /* lined up on the runway: its axis is the known heading, or the line-up heading (whose compass
     * error the wind filter later learns and removes, see fc.cpp) */
    ap->rwy_hdg = fc->cfg.rwy_hdg_deg >= 0.0f ? f_wrap(fc->cfg.rwy_hdg_deg * F_D2R) : y;
    ap->rwy_src = fc->cfg.rwy_hdg_deg >= 0.0f ? 2 : 0;
    ap->trk_n = ap->trk_e = 0.0f;
    ap->land_dir = y;
    ap->hdg_hold = y;
    ap->phase = 0;
    ap->armed = 1;
    go(ap, ST_TAKEOFF);
}

static void begin_land(FC *fc)
{
    choose_land_dir(fc);
    v3 p = est_pos(fc);
    float s, t;
    land_frame(fc, p, &s, &t);
    (void)s; (void)t;
    /* holding circle on the flying side of the runway, whichever way we land */
    int same = f_abs(f_wrap(fc->ap.land_dir - fc->ap.rwy_hdg)) < F_PI_2;
    fc->ap.wind_side = ((fc->cfg.fly_side > 0) == (same != 0)) ? 1 : 0;
    fc->ap.final_ok = 0;
    go(&fc->ap, ST_LAND);
}

static void dispatch(FC *fc)
{
    AP *ap = &fc->ap;
    uint8_t s = ap->state, g = ap->goal;
    if (g == FC_CMD_HOVER) {
        if (s == ST_LAND) go(ap, ST_TO_HOVER);
        else if (s == ST_FINAL) go(ap, ST_GO_AROUND);
    } else if (g == FC_CMD_LOOP) {
        if (s == ST_HOVER) { if (ap->exit_ok) go(ap, ST_FROM_HOVER); }
        else if (s == ST_CRUISE) go(ap, ST_LOOP_PREP);
        else if (s == ST_LAND) go(ap, ST_CRUISE);
        else if (s == ST_FINAL) go(ap, ST_GO_AROUND);
    } else if (g == FC_CMD_LAND) {
        if (s == ST_HOVER) { if (ap->exit_ok) go(ap, ST_FROM_HOVER); }
        else if (s == ST_CRUISE) begin_land(fc);
    } else if (g == FC_CMD_CRUISE) {
        if (s == ST_HOVER) { if (ap->exit_ok) go(ap, ST_FROM_HOVER); }
        else if (s == ST_LAND || s == ST_LOOP_PREP) go(ap, ST_CRUISE);
        else if (s == ST_FINAL) go(ap, ST_GO_AROUND);
    }
}

void ap_command(FC *fc, uint8_t cmd)
{
    AP *ap = &fc->ap;
    uint8_t s = ap->state;
    if (cmd == FC_CMD_NONE || s == ST_INIT || s == ST_MANUAL) return;
    int can_fly = fc->est.ready && fc->est.gps_ok;
    if (cmd == FC_CMD_TAKEOFF) {
        if (s == ST_GROUND && can_fly) { ap->goal = FC_CMD_NONE; begin_takeoff(fc); }
    } else if (cmd == FC_CMD_HOVER || cmd == FC_CMD_LOOP || cmd == FC_CMD_LAND) {
        if (s == ST_GROUND) {
            if (can_fly) { begin_takeoff(fc); ap->goal = cmd; }
        } else if (s != ST_ROLLOUT && s != ST_FLARE) {
            ap->goal = cmd;
            dispatch(fc);
        }
    } else if (cmd == FC_CMD_CRUISE) {
        if (s != ST_GROUND && s != ST_ROLLOUT && s != ST_FLARE) { ap->goal = cmd; dispatch(fc); }
    }
}

/* ------------------------------------------------------------------ manual (RC) */
static float stick(int16_t us) { return f_clamp(((float)us - 1500.0f) / 500.0f, -1.0f, 1.0f); }

static void manual(FC *fc, const fc_in_t *in, float u[3], float *thr, float *brake)
{
    float da = stick(in->rc_us[0]), de = stick(in->rc_us[1]), dr = stick(in->rc_us[3]);
    *thr = f_clamp(((float)in->rc_us[2] - 1000.0f) / 1000.0f, 0.0f, 1.0f);
    float V = fc->V;
    int on_gnd = est_alt(fc) < 0.35f && V < 7.0f;
    *brake = (*thr < 0.02f && fc->Vh < 2.0f) ? 1.0f : 0.0f;
    if (in->rc_mode == FC_RC_DIRECT || on_gnd) { u[0] = da; u[1] = de; u[2] = dr; return; }
    /* angle mode: sticks command roll / pitch angle; rudder is trim on top of auto-coordination */
    float phi = 1.0f * da;
    float theta = 0.07f + (de > 0.0f ? 0.55f * de : 0.45f * de);
    float tr = F_G * f_tan(phi) / f_max(V, 8.0f);
    fwd_att(fc, phi, theta, *thr, f_max(V, 3.0f), 0, 0.0f, tr, u);
    u[2] = f_clamp(u[2] + 0.5f * dr, -1.0f, 1.0f);
}

/* ------------------------------------------------------------------ main */
void ap_update(FC *fc, const fc_in_t *in, float dt)
{
    AP *ap = &fc->ap;
    const fc_config_t *c = &fc->cfg;
    ap->t_state += dt;
    v3 pos = est_pos(fc), vel = est_vel(fc);
    float V = fc->V, Vh = fc->Vh, alt = est_alt(fc);
    {
        m33 R = q_to_R(fc->est.q);
        R_to_euler(&R, &fc->yaw, &fc->pitch, &fc->roll);
    }
    float yaw = fc->yaw, pitch = fc->pitch, roll = fc->roll;
    ap->rwy_c = f_cos(ap->rwy_hdg); ap->rwy_s = f_sin(ap->rwy_hdg);
    float thr = 0.0f, brake = 0.0f;
    float u[3] = {0.0f, 0.0f, 0.0f};

    /* RC mode switch takes priority over everything */
    uint8_t rc = in->rc_mode;
    if (ap->state != ST_INIT) {
        if ((rc == FC_RC_DIRECT || rc == FC_RC_ASSIST) && ap->state != ST_MANUAL) {
            go(ap, ST_MANUAL);
            ap->goal = FC_CMD_NONE;
        } else if (!(rc == FC_RC_DIRECT || rc == FC_RC_ASSIST) && ap->state == ST_MANUAL) {
            /* hand back: airborne -> join the loiter circle, on the ground -> parked */
            ap->goal = FC_CMD_NONE;
            if (alt > 3.0f) { ap->alt_tgt = c->cruise_alt_m; go(ap, ST_CRUISE); }
            else go(ap, ST_GROUND);
        }
    }
    ap->prev_rc = rc;

    switch (ap->state) {
    case ST_INIT:
        brake = 1.0f;
        break;

    case ST_MANUAL:
        manual(fc, in, u, &thr, &brake);
        break;

    case ST_GROUND:
        brake = 1.0f;
        ap->hdg_hold = yaw;
        break;

    case ST_TAKEOFF: {
        thr = f_clamp(ap->t_state / 0.8f, 0.0f, 1.0f);
        /* runway axis from the raw GPS velocity of the first metres of the roll (the nose wheel runs
         * straight from the line-up there; the compass can be 10 deg off, and the fused velocity was
         * built with the compass heading) */
        if (ap->rwy_src == 0 && fc->gps_new && ap->phase == 0) {
            v3 g = fc->gps_v;
            float sp = f_sqrt(g.x * g.x + g.y * g.y);
            if (sp > 2.0f && sp < 8.5f) { ap->trk_n += g.x; ap->trk_e += g.y; }
        }
        /* steer the roll on GPS track back onto the runway axis once the track means something;
         * below that, hold the line-up yaw */
        float steer = ap->rwy_hdg;
        if (Vh > 4.0f) {
            float s_, xt;
            float want = line_heading(pos, ap->home[0], ap->home[1], ap->rwy_hdg, 20.0f, &s_, &xt);
            steer = yaw + f_clamp(f_wrap(want - course(fc)), -0.3f, 0.3f);
        }
        if (Vh < 7.5f && ap->phase == 0 && alt < 1.5f) {
            fwd_att(fc, 0.0f, 0.0f, thr, f_max(Vh, 3.0f), 1, steer, 0.0f, u);
        } else {
            if (ap->phase == 0 && ap->rwy_src == 0 && (ap->trk_n != 0.0f || ap->trk_e != 0.0f)) {
                float trk = f_atan2(ap->trk_e, ap->trk_n);
                if (f_abs(f_wrap(trk - ap->rwy_hdg)) < 0.35f) {
                    ap->rwy_hdg = trk; ap->land_dir = trk; ap->rwy_src = 1;
                    ap->rwy_c = f_cos(trk); ap->rwy_s = f_sin(trk);
                }
            }
            ap->phase = 1;
            float theta = f_clamp(0.14f + 0.05f * (V - 11.0f), 0.05f, 0.42f);
            thr = 0.8f;
            if (alt < 3.0f) {
                fwd_att(fc, 0.0f, theta, thr, f_max(V, 5.0f), 1, steer, 0.0f, u);
            } else {   /* climb out along the runway centreline: no drifting over the pilot in a crosswind */
                float s_, xt, phi, tr;
                float want = fence(fc, pos, line_heading(pos, ap->home[0], ap->home[1], ap->rwy_hdg, 25.0f, &s_, &xt), -2.0f);
                bank_from_heading(fc, want, V, 0.3f, &phi, &tr);
                fwd_att(fc, phi, theta, thr, f_max(V, 5.0f), 0, 0.0f, tr, u);
            }
            if (alt > 22.0f) {
                ap->phase = 0;
                go(ap, ST_CRUISE);
                dispatch(fc);
            }
        }
        break;
    }

    case ST_CRUISE: {
        float cn, ce;
        float orbit_t = (float)c->fly_side * (c->fence_m + ORBIT_R + 5.0f);
        rwy_point(ap, ORBIT_S, orbit_t, &cn, &ce);
        float hdg = fence(fc, pos, orbit_heading(pos, cn, ce, ORBIT_R, (float)c->fly_side), c->fence_m);
        float phi, tr, th;
        bank_side(fc, pos, hdg, V, 0.75f, &phi, &tr);
        tecs(fc, ap->alt_tgt, c->cruise_v, V, &th, &thr);
        int to_hover = ap->goal == FC_CMD_HOVER;
        int lvl = 0;
        if (to_hover) {
            /* turn to face the wind first: the hover then starts with the belly into the wind */
            float wn = fc->wind.x[0], we = fc->wind.x[1];
            if (fc->wind.ok && wn * wn + we * we > 1.5f * 1.5f && ap->t_state < 25.0f) {
                float into = f_atan2(-we, -wn);
                bank_side(fc, pos, fence(fc, pos, into, c->fence_m), V, 0.6f, &phi, &tr);
                lvl = f_abs(f_wrap(into - course(fc))) < 0.2f;
            } else lvl = 1;
            /* the pull-up and hover drift ~20 m along the flight path: start well inside the box */
            float ahead = side_dist(fc, v3_make(pos.x + 2.0f * vel.x, pos.y + 2.0f * vel.y, 0.0f));
            if (side_dist(fc, pos) < c->fence_m + 10.0f || ahead < c->fence_m + 10.0f) lvl = 0;
            if (lvl) { phi = 0.0f; tr = 0.0f; }
        }
        fwd_att(fc, phi, th, thr, V, 0, 0.0f, tr, u);
        if (to_hover && lvl) {
            if (ap->t_state > 1.0f && f_abs(roll) < 0.10f && f_abs(pitch) < 0.18f && f_abs(fc->est.w.x) < 0.25f
                && f_abs(vel.z) < 1.5f)
                go(ap, ST_TO_HOVER);
        } else if ((ap->goal == FC_CMD_LOOP || ap->goal == FC_CMD_LAND) && ap->t_state > 1.0f) {
            dispatch(fc);
        }
        break;
    }

    case ST_LOOP_PREP: {
        /* loops are flown on a line parallel to the runway on the flying side, in whichever direction
         * is nearer the current course, so the loop's vertical plane never reaches the pilot's side */
        float loop_t = (float)c->fly_side * (c->fence_m + 20.0f);
        if (ap->t_state < 0.015f) {
            float d = f_wrap(course(fc) - ap->rwy_hdg);
            ap->hdg_hold = f_abs(d) < F_PI_2 ? ap->rwy_hdg : f_wrap(ap->rwy_hdg + F_PI);
        }
        float th;
        tecs(fc, LOOP_ALT, 16.0f, V, &th, &thr);
        thr = f_clamp(thr + 0.15f, 0.0f, 1.0f);
        float pn, pe, along, xt;
        rwy_point(ap, LOOP_S, loop_t, &pn, &pe);
        float hdg = line_heading(pos, pn, pe, ap->hdg_hold, 30.0f, &along, &xt);
        if (along > 140.0f) {                               /* run out of field: come back the other way */
            ap->hdg_hold = f_wrap(ap->hdg_hold + F_PI);
            hdg = line_heading(pos, pn, pe, ap->hdg_hold, 30.0f, &along, &xt);
            ap->t_state = f_min(ap->t_state, 1.0f);
        }
        float phi, tr;
        bank_side(fc, pos, fence(fc, pos, hdg, c->fence_m), V, 0.75f, &phi, &tr);
        fwd_att(fc, phi, th, thr, V, 0, 0.0f, tr, u);
        int lined = f_abs(xt) < 6.0f && f_abs(f_wrap(course(fc) - ap->hdg_hold)) < 0.15f && along > -120.0f;
        int ready = lined && V > 15.0f && V < 18.5f && f_abs(roll) < 0.25f && f_abs(pitch) < 0.15f
                    && f_abs(alt - LOOP_ALT) < 3.0f && f_abs(vel.z) < 1.0f;
        if ((ready && ap->t_state > 1.5f) || (lined && ap->t_state > 20.0f)) {
            ap->loop_theta0 = pitch;
            ap->th_cmd = pitch;
            float hn = f_cos(ap->hdg_hold), he = f_sin(ap->hdg_hold);
            ap->gam_prev = f_atan2(-vel.z, vel.x * hn + vel.y * he);
            ap->gam_unw = ap->gam_prev;
            go(ap, ST_LOOP);
        }
        break;
    }

    case ST_LOOP: {
        float w = f_clamp(3.0f * F_G / f_max(V, 7.0f), 1.6f, 3.4f);
        thr = (ap->th_cmd - ap->loop_theta0 < 2.4f) ? 1.0f : 0.1f;     /* no extra energy on the way down */
        float th_end = ap->loop_theta0 + F_2PI;
        float hn = f_cos(ap->hdg_hold), he = f_sin(ap->hdg_hold);
        float gam = f_atan2(-vel.z, vel.x * hn + vel.y * he);
        ap->gam_unw += f_wrap(gam - ap->gam_prev);
        ap->gam_prev = gam;
        int done = ap->th_cmd >= th_end;
        if (!done) ap->th_cmd = f_min(f_min(ap->th_cmd + w * dt, ap->gam_unw + 0.30f), th_end);
        m33 Rd = euler_to_R(ap->hdg_hold, ap->th_cmd, 0.0f);
        v3 w_sp = att_ctrl(fc, &Rd, v3_make(0.0f, done ? 0.0f : w, 0.0f), 6.0f, 0, 0.0f);
        rate_ctrl(fc, w_sp, thr, V, BW_FWD, u);
        if (done && f_abs(roll) < 0.35f && f_abs(pitch) < 0.5f) {
            if (ap->goal == FC_CMD_LOOP) ap->goal = FC_CMD_NONE;
            go(ap, ST_CRUISE);
        }
        break;
    }

    case ST_TO_HOVER: {
        if (ap->t_state < 0.015f) { ap->hdg_hold = yaw; ap->th_cmd = pitch; ap->hover_int = 0.0f; }
        float hn = f_cos(ap->hdg_hold), he = f_sin(ap->hdg_hold);
        float gam = f_atan2(-vel.z, f_max(vel.x * hn + vel.y * he, 0.0f));
        float rate = f_clamp(2.0f * (F_PI_2 - pitch) + 0.12f, 0.12f, 1.4f);
        /* the nose may lead the flight path by only ~17 deg, else wing/tail stall and the elevator dies */
        ap->th_cmd = f_min(f_min(f_min(ap->th_cmd + rate * dt, gam + 0.30f), pitch + 0.35f), F_PI_2);
        m33 Rd = euler_to_R(ap->hdg_hold, ap->th_cmd, 0.0f);
        v3 w_ff = v3_make(0.0f, ap->th_cmd < F_PI_2 ? rate : 0.0f, 0.0f);
        /* full power for the pull-up; once the nose is high, below-hover thrust so gravity bleeds the
         * speed off (with T/W > 2 a full-power vertical climb would just keep going) */
        thr = pitch < 1.0f ? 1.0f : f_clamp(ap->thr_h - 0.04f * (V - 4.0f), 0.3f, 1.0f);
        v3 w_sp = att_ctrl(fc, &Rd, w_ff, 6.0f, 0, 0.0f);
        m33 R = q_to_R(fc->est.q);
        if (pitch > 1.0f) {
            rate_ctrl(fc, w_sp, thr, V, BW_HOVER, u);
            v3 vb = RT_mul(&R, vel);
            u[2] = f_clamp(u[2] - 0.14f * vb.y, -1.0f, 1.0f);
        } else {
            rate_ctrl(fc, w_sp, thr, V, BW_FWD, u);
        }
        if ((ap->th_cmd >= F_PI_2 - 1e-6f && pitch > 1.35f && (-R.m[2][0] > 0.99f || ap->t_state > 6.0f))
            || (ap->t_state > 8.0f && pitch > 1.0f && V < 8.0f)) {
            ap->hover_tgt[0] = pos.x + vel.x * 1.5f;
            ap->hover_tgt[1] = pos.y + vel.y * 1.5f;
            {   /* never hold the hover over the pilot's side */
                v3 tg = v3_make(ap->hover_tgt[0], ap->hover_tgt[1], 0.0f);
                float d = side_dist(fc, tg), need = c->fence_m + 5.0f;
                if (d < need) {
                    float k = (need - d) * (float)c->fly_side;
                    ap->hover_tgt[0] += -k * f_sin(ap->rwy_hdg);
                    ap->hover_tgt[1] += k * f_cos(ap->rwy_hdg);
                }
            }
            ap->alt_tgt = f_clamp(alt + f_max(-vel.z, 0.0f) * 1.5f, 25.0f, 60.0f);
            go(ap, ST_HOVER);
            ap->hover_int = 0.0f;
            ap->nose_h_f[0] = ap->nose_h_f[1] = 0.0f;
            ap->th_cmd = F_PI_2;
        }
        break;
    }

    case ST_HOVER: {
        float an = 0.25f * (ap->hover_tgt[0] - pos.x) - 0.9f * vel.x;
        float ae = 0.25f * (ap->hover_tgt[1] - pos.y) - 0.9f * vel.y;
        float na = f_sqrt(an * an + ae * ae);
        if (na > 5.0f) { an *= 5.0f / na; ae *= 5.0f / na; }       /* up to ~27 deg of lean into the wind */
        float tn = an / F_G, te = ae / F_G;
        float nz = f_sqrt(f_max(1.0f - tn * tn - te * te, 0.5f));
        v3 n = v3_make(tn, te, -nz);
        n = v3_scale(n, 1.0f / v3_norm(n));
        m33 Rd = R_from_nose(n, ap->hdg_hold);
        float vz_t = f_clamp(0.5f * (ap->alt_tgt - alt), -2.0f, 2.0f);
        float ez = vz_t - (-vel.z);
        ap->hover_int = f_clamp(ap->hover_int + 0.04f * ez * dt, -0.2f, 0.2f);
        thr = f_clamp(ap->thr_h + 0.12f * ez + ap->hover_int, 0.4f, 1.0f);
        if (-vel.z > 2.0f) thr = f_max(thr, ap->thr_h - 0.03f);   /* keep prop wash, but below hover thrust */
        v3 w_sp = att_ctrl(fc, &Rd, v3_make(0.0f, 0.0f, 0.0f), 6.0f, 0, 0.0f);
        rate_ctrl(fc, w_sp, thr, V, BW_HOVER, u);
        /* wind leans the nose upwind; slowly turn the belly to face along the wind line */
        m33 R = q_to_R(fc->est.q);
        /* hover abort: blown towards the pilot's side or far off its spot -> fly out of it */
        {
            float dn = pos.x - ap->hover_tgt[0], de = pos.y - ap->hover_tgt[1];
            if (side_dist(fc, pos) < c->fence_m + 5.0f || dn * dn + de * de > 30.0f * 30.0f)
                if (ap->goal == FC_CMD_HOVER || ap->goal == FC_CMD_NONE) { ap->goal = FC_CMD_CRUISE; ap->t_state = f_max(ap->t_state, 1.6f); }
        }
        int leaving = (ap->goal == FC_CMD_LOOP || ap->goal == FC_CMD_LAND || ap->goal == FC_CMD_CRUISE) && ap->t_state > 1.5f;
        ap->nose_h_f[0] += (dt / 0.8f) * (R.m[0][0] - ap->nose_h_f[0]);
        ap->nose_h_f[1] += (dt / 0.8f) * (R.m[1][0] - ap->nose_h_f[1]);
        if (!leaving && f_sqrt(ap->nose_h_f[0] * ap->nose_h_f[0] + ap->nose_h_f[1] * ap->nose_h_f[1]) > 0.06f) {
            float ang = f_atan2(ap->nose_h_f[1], ap->nose_h_f[0]);
            float d = f_wrap(ang - ap->hdg_hold);
            if (f_abs(d) > F_PI_2) d = f_wrap(d - F_PI);
            ap->hdg_hold = f_wrap(ap->hdg_hold + f_clamp(1.5f * d, -1.2f, 1.2f) * dt);
        }
        v3 vb = RT_mul(&R, vel);
        u[2] = f_clamp(u[2] - 0.14f * vb.y, -1.0f, 1.0f);
        if (leaving) {
            /* turn the belly to a safe exit heading first, then go */
            float want = safe_exit(fc, ap->hdg_hold);
            float d = f_wrap(want - ap->hdg_hold);
            ap->hdg_hold = f_wrap(ap->hdg_hold + f_clamp(d, -1.0f * dt, 1.0f * dt));
            float belly = f_atan2(R.m[1][2], R.m[0][2]);
            int aligned = f_abs(d) < 0.02f && f_abs(f_wrap(belly - ap->hdg_hold)) < 0.35f;
            if (aligned || ap->t_state > 1.5f + 4.0f + 3.2f) {
                ap->thr_h = f_clamp(ap->thr_h + ap->hover_int, 0.3f, 0.9f);   /* remember this plane's hover throttle */
                ap->hover_int = 0.0f;
                ap->hdg_hold = safe_exit(fc, ap->hdg_hold);
                ap->exit_ok = 1;
                dispatch(fc);
                ap->exit_ok = 0;
            }
        }
        break;
    }

    case ST_FROM_HOVER: {
        if (ap->t_state < 0.015f) ap->th_cmd = pitch;
        ap->hdg_hold = fence(fc, pos, ap->hdg_hold, c->fence_m);   /* never dive out towards the pilot */
        const float rate = 0.9f;
        ap->th_cmd = f_max(ap->th_cmd - rate * dt, 0.30f);
        m33 Rd = euler_to_R(ap->hdg_hold, ap->th_cmd, 0.0f);
        thr = 1.0f;
        v3 w_sp = att_ctrl(fc, &Rd, v3_make(0.0f, ap->th_cmd > 0.30f ? -rate : 0.0f, 0.0f), 6.0f, 0, 0.0f);
        rate_ctrl(fc, w_sp, thr, V, BW_FWD, u);
        /* airspeed, not ground speed: into a headwind the ground speed may never get there */
        if ((ap->th_cmd <= 0.31f && V > 10.0f) || ap->t_state > 8.0f) {
            ap->alt_tgt = c->cruise_alt_m;
            go(ap, ST_CRUISE);
        }
        break;
    }

    case ST_LAND: {
        /* holding circle beside the extended centreline, tangent to it at s = HOLD_S */
        float on, oe;
        land_origin(fc, &on, &oe);
        float side = ap->wind_side ? 1.0f : -1.0f;          /* +1 circle right of the centreline */
        float lc = f_cos(ap->land_dir), ls = f_sin(ap->land_dir);
        const float HOLD_R = c->hold_r_m, HOLD_S = hold_s(c), HOLD_OFF = 8.0f;
        /* the circle stays HOLD_OFF clear of the centreline; final is joined with a short intercept */
        float cn = on + HOLD_S * lc - side * (HOLD_R + HOLD_OFF) * ls;
        float ce = oe + HOLD_S * ls + side * (HOLD_R + HOLD_OFF) * lc;
        float dir = side;                                   /* right-hand circle flown clockwise */
        float hdg = fence(fc, pos, orbit_heading(pos, cn, ce, HOLD_R, dir), 3.0f);
        int high = alt > c->pattern_alt_m + 10.0f;
        if (high) {   /* far too high: spiral down in the loiter orbit, well out on the flying side */
            float on2, oe2;
            rwy_point(ap, ORBIT_S, (float)c->fly_side * (c->fence_m + ORBIT_R + 5.0f), &on2, &oe2);
            hdg = fence(fc, pos, orbit_heading(pos, on2, oe2, ORBIT_R, (float)c->fly_side), c->fence_m);
            ap->t_state = 0.0f;
        }
        float phi, tr, th;
        bank_side(fc, pos, hdg, V, 0.6f, &phi, &tr);
        float alt_t = c->pattern_alt_m;
        float s, t;
        land_frame(fc, pos, &s, &t);
        tecs(fc, alt_t, c->cruise_v - 1.0f, V, &th, &thr);
        {   /* spiral down (shed height, not speed) only once established in the circle */
            float ocn = cn, oce = ce, orr = HOLD_R;
            if (high) {
                rwy_point(ap, ORBIT_S, (float)c->fly_side * (c->fence_m + ORBIT_R + 5.0f), &ocn, &oce);
                orr = ORBIT_R;
            }
            float dd = f_sqrt((pos.x - ocn) * (pos.x - ocn) + (pos.y - oce) * (pos.y - oce));
            if (alt > alt_t + 4.0f && dd < 1.6f * orr) thr = high ? 0.0f : f_min(thr, 0.15f);
        }
        fwd_att(fc, phi, th, thr, V, 0, 0.0f, tr, u);
        float dn = pos.x - cn, de = pos.y - ce;
        int on_circle = !high && f_abs(f_sqrt(dn * dn + de * de) - HOLD_R) < 0.5f * HOLD_R;
        int lined_up = f_abs(f_wrap(course(fc) - ap->land_dir)) < 0.2f && f_abs(t) < 0.3f * HOLD_R + HOLD_OFF
                       && s > HOLD_S - 50.0f && s < HOLD_S + 25.0f;
        if (on_circle && lined_up && f_abs(alt - alt_t) < 3.0f && ap->t_state > 5.0f) {
            ap->glide_int = 0.0f;
            go(ap, ST_FINAL);
        }
        break;
    }

    case ST_FINAL: {
        float on, oe;
        land_origin(fc, &on, &oe);
        float s, t;
        float hdg = fence(fc, pos, line_heading(pos, on, oe, ap->land_dir, alt > 5.0f ? 35.0f : 25.0f, &s, &t), -1.0f);
        float tg = f_tan(c->glide_deg * F_D2R);
        float h_gp = f_clamp(-s * tg, 0.0f, c->pattern_alt_m);
        float gs = vel.x * f_cos(ap->land_dir) + vel.y * f_sin(ap->land_dir);
        float vz_gp = (h_gp > 0.0f && h_gp < c->pattern_alt_m) ? gs * tg : 0.0f;   /* sink along the path */
        float phi, tr;
        bank_from_heading(fc, hdg, V, alt > 10.0f ? 0.35f : 0.18f, &phi, &tr);
        /* approach speed + half the headwind (wind shear and gusts on short final) */
        float head = -(fc->wind.x[0] * f_cos(ap->land_dir) + fc->wind.x[1] * f_sin(ap->land_dir));
        float Vt = c->v_approach + f_clamp(0.5f * head, 0.0f, 3.0f);
        /* pitch holds airspeed (back side of the drag curve is flown with the throttle) */
        float th = f_clamp(0.06f + 0.06f * (V - Vt), -0.20f, 0.25f);
        float eh = h_gp - alt;
        float esink = vz_gp - vel.z;                         /* +: sinking slower than the path */
        ap->glide_int = f_clamp(ap->glide_int + 0.02f * eh * dt, -0.2f, 0.3f);
        thr = f_clamp(0.30f + ap->glide_int + 0.08f * eh - 0.15f * esink, 0.0f, 1.0f);
        fwd_att(fc, phi, th, thr, V, 0, 0.0f, tr, u);
        /* stabilised-approach gate */
        int unstable = (alt < 0.45f * c->pattern_alt_m) && (f_abs(t) > 8.0f || f_abs(f_wrap(course(fc) - ap->land_dir)) > 0.5f
                                         || V < Vt - 3.0f || V > Vt + 5.0f);
        float rwy_left = (fc->cfg.rwy_len_ahead_m + fc->cfg.rwy_len_behind_m) - AIM_IN;   /* s at far end */
        int long_ = s > 0.5f * rwy_left && alt > 1.5f;
        if ((unstable || long_) && ap->ga_count < 3) {
            ap->ga_count++;
            go(ap, ST_GO_AROUND);
        } else if (alt < c->flare_h_m && s > -60.0f) {
            ap->touch_t = 0.0f;
            go(ap, ST_FLARE);
        }
        break;
    }

    case ST_FLARE: {
        float on, oe, s, t;
        land_origin(fc, &on, &oe);
        float hdg = line_heading(pos, on, oe, ap->land_dir, 15.0f, &s, &t);
        (void)hdg;
        float h = f_max(alt, 0.0f);
        float vz_t = f_clamp(0.35f + h / c->flare_tau_s, 0.35f, 1.5f);   /* target sink, m/s (down +) */
        float th = f_clamp(0.05f + 0.15f * (vel.z - vz_t), -0.03f, 0.14f);   /* tail strikes at ~12 deg */
        thr = (vel.z > vz_t + 0.8f && h > 0.8f) ? 0.25f : 0.0f;   /* a touch of power if dropping */
        fwd_att(fc, 0.0f, th, thr, f_max(V, 6.0f), 1, ap->land_dir - f_clamp(t / 30.0f, -0.08f, 0.08f), 0.0f, u);
        /* touchdown: wheel impact, or settled on the runway */
        if (fc->est.acc.z < -1.3f * F_G && h < 0.6f) ap->touch_t += 1.0f;   /* wheel impact (not the flare pull) */
        /* settled: no sink for 0.6 s while within baro/GPS height error of the ground */
        if (h < 1.2f && f_abs(vel.z) < 0.2f) ap->touch_t += dt; else if (ap->touch_t < 1.0f) ap->touch_t = 0.0f;
        if ((ap->touch_t > 0.6f && ap->t_state > 0.3f) || ap->t_state > 10.0f) {
            ap->slow_t = 0.0f;
            go(ap, ST_ROLLOUT);
        }
        break;
    }

    case ST_ROLLOUT: {
        float on, oe, s, t;
        land_origin(fc, &on, &oe);
        float hdg = line_heading(pos, on, oe, ap->land_dir, 12.0f, &s, &t);
        thr = 0.0f;
        brake = c->brake_enable ? f_clamp((ap->t_state - 0.6f) * 1.5f, 0.0f, 0.8f) : 0.0f;
        fwd_att(fc, 0.0f, -0.02f, thr, f_max(Vh, 3.0f), 1, hdg, 0.0f, u);
        ap->slow_t = (Vh < 0.35f) ? ap->slow_t + dt : 0.0f;
        if (ap->slow_t > 1.0f) {
            ap->goal = FC_CMD_NONE;
            ap->ga_count = 0;
            ap->armed = 0;
            go(ap, ST_GROUND);
        }
        break;
    }

    case ST_GO_AROUND: {
        thr = 1.0f;
        float th = 0.22f;
        float on, oe, s, t;
        land_origin(fc, &on, &oe);
        float hdg = fence(fc, pos, line_heading(pos, on, oe, ap->land_dir, 40.0f, &s, &t), -5.0f);
        float phi, tr;
        bank_from_heading(fc, hdg, V, alt > 8.0f ? 0.4f : 0.1f, &phi, &tr);
        fwd_att(fc, phi, th, thr, V, 0, 0.0f, tr, u);
        if (alt > c->pattern_alt_m - 4.0f) {
            if (ap->goal == FC_CMD_LAND || ap->goal == FC_CMD_NONE) { ap->goal = FC_CMD_LAND; begin_land(fc); }
            else { go(ap, ST_CRUISE); dispatch(fc); }
        }
        break;
    }
    }

    ap->out[0] = u[0]; ap->out[1] = u[1]; ap->out[2] = u[2];
    ap->out[3] = thr; ap->out[4] = brake;
}
