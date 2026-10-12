"""Button-driven autopilot: TAKEOFF / HOVER / LOOP / LAND on top of CRUISE (orbit loiter).

Inputs come only from the Estimator (noisy sensors).  Outputs are normalised moment commands
da/de/dr in [-1,1], throttle 0..1 and a wheel brake.
Attitude control is quaternion-error based (no gimbal lock) so the same code flies hover and loops.

Buttons:  takeoff | hover | loop | land | cruise
A button pressed in the "wrong" state chains the needed steps (e.g. land from hover = transition,
join approach, land).
"""
import numpy as np
from .mathutil import qmul, qconj, R2q, R2euler, euler2R, R_from_nose, wrap

G = 9.80665
RHO = 1.225
RWY_HDG = 0.0                          # runway points north from the origin
ORBIT_C = np.array([90.0, 70.0])       # loiter circle centre (N, E)
ORBIT_R = 60.0
CRUISE_ALT, CRUISE_V = 28.0, 13.0
LOOP_ALT = 50.0
# nominal control power (moment per Pa per command / inertia) used for gain scheduling
AUTH_Q = np.array([3.6, 2.5, 1.8])         # rad/s^2 per Pa dynamic pressure per unit command
AUTH_W = np.array([85.0, 168.0, 110.0])    # rad/s^2 per unit command in a hover (T = weight)


def clip(x, a, b):
    return max(a, min(b, x))


class Autopilot:
    def __init__(self):
        self.state = 'GROUND'
        self.goal = None
        self.t_state = 0.0
        self.nav_ticks = 0
        self.int_rate = np.zeros(3)
        self.thr_h = 0.55
        self.hover_int = 0.0
        self.th_cmd = 0.0
        self.hdg_hold = 0.0
        self.hover_tgt = np.zeros(3)
        self.alt_tgt = CRUISE_ALT
        self.loop_theta0 = 0.0
        self.log_msg = ''
        self.out = dict(da=0.0, de=0.0, dr=0.0, thr=0.0, brake=1.0)
        self.slip_f = 0.0
        self.touch_t = 0.0
        self.final = False

    # ------------------------------------------------------------------ buttons
    def command(self, name):
        s = self.state
        if name == 'takeoff':
            if s == 'GROUND':
                self.goal = None
                self._go('TAKEOFF')
        elif name in ('hover', 'loop', 'land'):
            if s == 'GROUND':
                self._go('TAKEOFF')
                self.goal = name
            elif s in ('ROLLOUT', 'FLARE'):
                pass
            else:
                self.goal = name
                self._dispatch()
        elif name == 'cruise':
            self.goal = 'cruise'
            self._dispatch()

    def _go(self, s):
        self.state = s
        self.t_state = 0.0
        self.int_rate[:] = 0

    def _dispatch(self):
        """Pick the next state to head towards self.goal from the current (airborne) state."""
        s, g = self.state, self.goal
        if g == 'hover':
            if s in ('CRUISE', 'LAND'):
                self._begin_hover_pull()
            elif s == 'LOOP' or s == 'TAKEOFF' or s == 'FROM_HOVER':
                pass                                       # will dispatch when they finish
        elif g == 'loop':
            if s == 'HOVER':
                self._go('FROM_HOVER')
            elif s in ('CRUISE',):
                self._go('LOOP_PREP')
        elif g == 'land':
            if s == 'HOVER':
                self._go('FROM_HOVER')
            elif s == 'CRUISE':
                self._go('LAND')
        elif g == 'cruise':
            if s == 'HOVER':
                self._go('FROM_HOVER')
            elif s in ('LAND', 'LOOP_PREP'):
                self._go('CRUISE')

    def _begin_hover_pull(self, est=None):
        self._go('TO_HOVER')

    def _finished_state(self):
        """Called when a state ends in CRUISE or HOVER so a queued goal continues."""
        self._dispatch()

    # ------------------------------------------------------------------ helpers
    def authority(self, V, thr):
        """Estimated control power (rad/s^2 per unit command) = airspeed part + prop-wash part
        (numbers from ident.py: measured on the physics model at hover and 13 m/s)."""
        T = 25.0 * thr * thr * max(1.0 - V / 28.0, 0.0)
        return AUTH_Q * (0.5 * RHO * V * V) + AUTH_W * (T / 8.3)

    def q_eff_est(self, V, thr):
        return 1.0

    def att_ctrl(self, est, R_des, w_ff, thr, V, yaw_gain=6.0, yaw_rate_sp=None):
        q_e = est.q
        q_d = R2q(R_des)
        qerr = qmul(qconj(q_e), q_d)
        if qerr[0] < 0:
            qerr = -qerr
        e = 2 * qerr[1:]
        Kp = np.array([8.0, 8.0, yaw_gain])
        w_sp = np.clip(Kp * e, -np.array([9, 9, 4.0]), np.array([9, 9, 4.0])) + w_ff
        if yaw_rate_sp is not None:
            w_sp[2] = yaw_rate_sp
        return w_sp

    def rate_ctrl(self, est, w_sp, thr, V, dt):
        auth = np.maximum(self.authority(V, thr), 12.0)
        bw = np.array([16.0, 14.0, 14.0])
        Kr = bw / auth
        err = w_sp - est.w
        # integrator trims out steady torques (prop torque, slipstream swirl); light leak against windup
        self.int_rate = np.clip(self.int_rate * (1 - 0.3 * dt) + err * dt * 4.0 * Kr, -0.4, 0.4)
        u = Kr * err + self.int_rate
        return np.clip(u, -1, 1)

    def fwd_att(self, est, phi, theta, thr, V, dt, hdg_hold=None, turn_rate=0.0):
        """Conventional attitude hold: tracks roll/pitch, yaw follows (coordinated) or holds heading."""
        yaw, _, _ = est.euler()
        R_des = euler2R(yaw, theta, phi)
        psid = turn_rate
        q_ff = psid * np.sin(phi)
        r_ff = psid * np.cos(phi)
        w_ff = np.array([0.0, q_ff, 0.0])
        if hdg_hold is not None:
            r_sp = 3.0 * wrap(hdg_hold - yaw)
        else:
            r_sp = r_ff
        w_sp = self.att_ctrl(est, R_des, w_ff, thr, V, yaw_rate_sp=r_sp)
        u = self.rate_ctrl(est, w_sp, thr, V, dt)
        self.slip_f += 0.05 * (est.acc[1] - self.slip_f)
        u[2] = clip(u[2] - 0.06 * self.slip_f, -1, 1)
        if self.state == 'TAKEOFF' and hdg_hold is not None:
            u[2] = clip(u[2] + 0.18 * thr, -1, 1)      # right-rudder for prop torque / swirl on the roll
        return u

    # ------------------------------------------------------------------ guidance
    def orbit_heading(self, pos):
        d = pos[:2] - ORBIT_C
        dist = np.linalg.norm(d)
        beta = np.arctan2(d[1], d[0])
        tangent = beta + np.pi / 2              # clockwise
        L1 = 30.0
        hdg = tangent + np.arctan((dist - ORBIT_R) / L1)
        return wrap(hdg), np.arctan(CRUISE_V ** 2 / (ORBIT_R * G))

    def line_heading(self, pos, p0, hdg_line, L1=25.0):
        d = pos[:2] - p0
        along = d[0] * np.cos(hdg_line) + d[1] * np.sin(hdg_line)
        xt = -d[0] * np.sin(hdg_line) + d[1] * np.cos(hdg_line)   # +ve = right of line
        hdg = hdg_line - np.arctan(xt / L1) * 1.0
        hdg = hdg_line + clip(wrap(hdg - hdg_line), -1.0, 1.0)
        return wrap(hdg), along, xt

    def bank_from_heading(self, est, hdg_des, V, phi_ff=0.0):
        vel = est.vel
        if np.hypot(vel[0], vel[1]) > 2.0:
            course = np.arctan2(vel[1], vel[0])
        else:
            course = est.euler()[0]
        err = wrap(hdg_des - course)
        phi = clip(1.4 * err + phi_ff * np.sign(err) * 0.0, -0.75, 0.75)
        turn_rate = G * np.tan(phi) / max(V, 6.0)
        return phi, turn_rate

    def tecs(self, est, alt_t, V_t, V, kp_alt=0.08):
        alt = est.alt
        vz = -est.vel[2]
        pitch_trim = 0.06
        vz_t = clip(0.5 * (alt_t - alt), -2.0, 3.0)
        theta = pitch_trim + kp_alt * 1.0 * (vz_t - vz) + 0.03 * (vz_t)
        theta = clip(theta + 0.04 * (V_t - V) * 0.0, -0.35, 0.40)
        thr = clip(0.52 + 0.10 * (V_t - V) + 0.03 * (vz_t - vz), 0.05, 1.0)
        return theta, thr

    # ------------------------------------------------------------------ main
    def update(self, est, dt):
        self.t_state += dt
        pos, vel = est.pos, est.vel
        V = float(np.linalg.norm(vel))
        Vh = float(np.hypot(vel[0], vel[1]))
        alt = est.alt
        yaw, pitch, roll = est.euler()
        s = self.state
        o = self.out
        thr = 0.0
        brake = 0.0
        u = np.zeros(3)

        if s == 'GROUND':
            thr, brake = 0.0, 1.0
            self.hdg_hold = yaw

        elif s == 'TAKEOFF':
            if self.t_state < 0.1:
                self.hdg_hold = RWY_HDG
            thr = clip(self.t_state / 0.8, 0.0, 1.0)
            if Vh < 7.5 and self.out.get('phase', 0) == 0 and alt < 1.5:
                u = self.fwd_att(est, 0.0, 0.0, thr, max(Vh, 3), dt, hdg_hold=self.hdg_hold)
            else:
                self.out['phase'] = 1
                Vc = 11.0
                theta = clip(0.14 + 0.05 * (Vh - Vc), 0.05, 0.42)
                hold = self.hdg_hold if alt < 6 else None
                thr = 0.8
                u = self.fwd_att(est, 0.0, theta, thr, max(Vh, 5), dt, hdg_hold=hold)
                if alt > 22.0:
                    self.out['phase'] = 0
                    self._go('CRUISE')
                    self._finished_state()

        elif s == 'CRUISE':
            hdg, _ = self.orbit_heading(pos)
            phi, tr = self.bank_from_heading(est, hdg, V)
            th, thr = self.tecs(est, CRUISE_ALT, CRUISE_V, V)
            u = self.fwd_att(est, phi, th, thr, V, dt, turn_rate=tr)
            if self.goal in ('hover', 'loop', 'land') and self.t_state > 1.0:
                self._dispatch()

        elif s == 'LOOP_PREP':
            if self.t_state < 0.05:
                self.hdg_hold = np.arctan2(vel[1], vel[0])
            th, thr = self.tecs(est, LOOP_ALT, 16.0, V)
            thr = clip(thr + 0.15, 0.0, 1.0)
            dC = ORBIT_C - pos[:2]
            far = np.hypot(*dC) > 110
            if far:                           # steer back towards the field before looping
                phi, tr = self.bank_from_heading(est, np.arctan2(dC[1], dC[0]), V)
                u = self.fwd_att(est, phi, th, thr, V, dt, turn_rate=tr)
                self.hdg_hold = np.arctan2(vel[1], vel[0])
                self.t_state = min(self.t_state, 1.0)
            else:
                u = self.fwd_att(est, 0.0, th, thr, V, dt, hdg_hold=self.hdg_hold)
            ready = (not far) and (V > 15.0 and V < 18.5 and abs(roll) < 0.25 and abs(pitch) < 0.15
                     and abs(alt - LOOP_ALT) < 3.0 and abs(vel[2]) < 1.0)
            if (ready and self.t_state > 1.5) or self.t_state > 12.0:
                self.loop_theta0 = pitch
                self.th_cmd = pitch
                hh = np.array([np.cos(self.hdg_hold), np.sin(self.hdg_hold)])
                self.gam_prev = np.arctan2(-vel[2], vel[:2] @ hh)
                self.gam_unw = self.gam_prev
                self._go('LOOP')

        elif s == 'LOOP':
            w = clip(3.0 * G / max(V, 7.0), 1.6, 3.4)
            thr = 1.0 if self.th_cmd - self.loop_theta0 < 2.4 else 0.1     # no extra energy on the way down
            th_end = self.loop_theta0 + 2 * np.pi
            # unwrapped flight-path angle in the loop plane: nose may lead it by at most ~0.3 rad
            hh = np.array([np.cos(self.hdg_hold), np.sin(self.hdg_hold)])
            gam = np.arctan2(-vel[2], vel[:2] @ hh)
            self.gam_unw += wrap(gam - self.gam_prev)
            self.gam_prev = gam
            done = self.th_cmd >= th_end
            if not done:
                self.th_cmd = min(self.th_cmd + w * dt, self.gam_unw + 0.30, th_end)
            R_des = euler2R(self.hdg_hold, self.th_cmd, 0.0)
            w_sp = self.att_ctrl(est, R_des, np.array([0.0, 0.0 if done else w, 0.0]), thr, V)
            u = self.rate_ctrl(est, w_sp, thr, V, dt)
            if done and abs(roll) < 0.35 and abs(pitch) < 0.5:
                self.goal = None if self.goal == 'loop' else self.goal
                self._go('CRUISE')

        elif s == 'TO_HOVER':
            if self.t_state < 0.05:
                self.hdg_hold = yaw             # pull up in the plane's own vertical plane
                self.th_cmd = pitch
                self.hover_int = 0.0
            hh = np.array([np.cos(self.hdg_hold), np.sin(self.hdg_hold)])
            gam = np.arctan2(-vel[2], max(vel[:2] @ hh, 0.0))               # climb angle of the flight path
            rate = clip(2.0 * (np.pi / 2 - pitch) + 0.12, 0.12, 1.4)    # ease into vertical, no overshoot
            # the nose may lead the flight path by only ~17 deg, otherwise wing/tail stall and the elevator dies
            self.th_cmd = min(self.th_cmd + rate * dt, gam + 0.30, pitch + 0.35, np.pi / 2)
            R_des = euler2R(self.hdg_hold, self.th_cmd, 0.0)
            w_ff = np.array([0.0, rate if self.th_cmd < np.pi / 2 else 0.0, 0.0])
            thr = 1.0 if (pitch < 1.3 or V > 6.0) else self.thr_h      # a slow prop makes no thrust at 16 m/s
            w_sp = self.att_ctrl(est, R_des, w_ff, thr, V)
            u = self.rate_ctrl(est, w_sp, thr, V, dt)
            if self.th_cmd >= np.pi / 2 - 1e-6 and pitch > 1.35:
                self.hover_tgt = np.array([pos[0] + vel[0] * 1.5, pos[1] + vel[1] * 1.5, 0.0])
                self.alt_tgt = 35.0
                self._go('HOVER')
                self.hover_int = 0.0
                self.th_cmd = np.pi / 2

        elif s == 'HOVER':
            # horizontal: position -> accel -> nose tilt
            ep = self.hover_tgt[:2] - pos[:2]
            a_cmd = 0.25 * ep - 0.9 * vel[:2]
            amax = 3.0
            na = np.linalg.norm(a_cmd)
            if na > amax:
                a_cmd *= amax / na
            tilt = a_cmd / G
            nz = np.sqrt(max(1 - tilt @ tilt, 0.5))
            n = np.array([tilt[0], tilt[1], -nz])
            n /= np.linalg.norm(n)
            R_des = R_from_nose(n, self.hdg_hold)
            thr_ff = self.thr_h
            vz_t = clip(0.5 * (self.alt_tgt - alt), -2.0, 2.0)
            ez = vz_t - (-vel[2])
            self.hover_int = clip(self.hover_int + 0.04 * ez * dt, -0.2, 0.2)
            thr = clip(thr_ff + 0.12 * ez + self.hover_int, 0.2, 1.0)
            if -vel[2] > 2.0:
                thr = max(thr, 0.65)          # shedding climb speed: keep prop wash over the tail for authority
            w_sp = self.att_ctrl(est, R_des, np.zeros(3), thr, V, yaw_gain=6.0)
            u = self.rate_ctrl(est, w_sp, thr, V, dt)
            if self.goal in ('loop', 'land', 'cruise') and self.t_state > 1.5:
                self._dispatch()

        elif s == 'FROM_HOVER':
            if self.t_state < 0.05:
                self.th_cmd = pitch
            rate = 0.9
            self.th_cmd = max(self.th_cmd - rate * dt, 0.30)
            R_des = euler2R(self.hdg_hold, self.th_cmd, 0.0)
            thr = 1.0
            w_sp = self.att_ctrl(est, R_des, np.array([0.0, -rate if self.th_cmd > 0.30 else 0.0, 0.0]), thr, V)
            u = self.rate_ctrl(est, w_sp, thr, V, dt)
            if self.th_cmd <= 0.31 and Vh > 9.5:
                self.alt_tgt = CRUISE_ALT
                self._go('CRUISE')

        elif s == 'LAND':
            p0 = np.array([-210.0, 0.0])
            hdg, along, xt = self.line_heading(pos, p0, RWY_HDG)
            s_go = -pos[0]                       # distance remaining to touchdown (north is track)
            if self.t_state < 0.05:
                self.final = False
            course = np.arctan2(vel[1], vel[0])
            if not self.final and (np.hypot(*(pos[:2] - p0)) < 35 or
                                   (abs(xt) < 20 and -215 < pos[0] < -120 and abs(wrap(course - RWY_HDG)) < 0.5)):
                self.final = True
            on_line = self.final
            Vt = 9.5
            if s_go > 150 or not on_line:
                alt_t = 18.0
            else:
                alt_t = max(-1.5, s_go * 0.115)
            if not on_line:
                hdg = np.arctan2(p0[1] - pos[1], p0[0] - pos[0])      # head for the approach start
            phi, tr = self.bank_from_heading(est, hdg, V)
            if on_line:
                phi = clip(phi, -0.3, 0.3) if alt > 10 else clip(phi, -0.12, 0.12)
            if on_line and s_go < 150:
                # final: pitch holds airspeed, throttle holds the glide path
                th = clip(0.07 + 0.07 * (V - Vt), -0.25, 0.30)
                thr = clip(0.18 + 0.10 * (alt_t - alt) + 0.15 * (-V * 0.115 - (-vel[2])), 0.0, 1.0)
            else:
                th, thr = self.tecs(est, alt_t, Vt, V)
            if on_line and alt < 1.3:
                self.out['phase'] = 0
                self._go('FLARE')
            u = self.fwd_att(est, phi, th, thr, V, dt, turn_rate=tr)

        elif s == 'FLARE':
            # idle, wings level, gentle nose-up so the mains touch first; altitude is only +-1 m so
            # the glide slope itself delivers the aircraft to the ground at ~1 m/s sink
            thr = 0.0
            theta = 0.07
            u = self.fwd_att(est, 0.0, theta, thr, max(V, 6), dt, hdg_hold=RWY_HDG)
            if self.t_state > 0.4 and est.acc[2] < -G * 1.5:
                self.touch_t = self.t_state
                self._go('ROLLOUT')

        elif s == 'ROLLOUT':
            thr = 0.0
            brake = 0.7 if self.t_state > 0.8 else 0.0
            u = self.fwd_att(est, 0.0, -0.02, thr, max(Vh, 3), dt, hdg_hold=RWY_HDG)
            if Vh < 0.4 and self.t_state > 2.0:
                self.goal = None
                self._go('GROUND')

        o['da'], o['de'], o['dr'] = float(u[0]), float(u[1]), float(u[2])
        o['thr'], o['brake'] = float(thr), float(brake)
        return o
