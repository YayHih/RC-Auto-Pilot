"""State estimator: Mahony MARG attitude filter + 3x(pos,vel,accel-bias) Kalman filters.

Uses only IMU, compass, baro and GPS - nothing the sim knows privately.
Attitude uses the accelerometer only when |f| is close to 1 g (unaccelerated), so it rides
through loops on the gyro and re-converges afterwards.
"""
import numpy as np
from .mathutil import qmul, q2R, R2q, R2euler, euler2R
from .sensors import MAG_NED, DECL, MOTOR_FIELD

G = 9.80665


class Estimator:
    def __init__(self):
        self.q = np.array([1.0, 0, 0, 0])
        self.eint = np.zeros(3)
        self.Kp_acc, self.Ki, self.Kp_mag = 2.0, 0.03, 0.8
        self.x = np.zeros((3, 3))          # per axis [p, v, accel bias]
        self.P = np.tile(np.diag([4.0, 1.0, 0.05]), (3, 1, 1))
        self.w = np.zeros(3)
        self.acc = np.array([0, 0, -G])
        self.a_ned_f = np.zeros(3)
        self.alt_baro0 = 0.0
        self.thr = 0.0
        self.m_ref = MAG_NED / np.linalg.norm(MAG_NED)
        self.gps_ok = False
        self.a_gps = np.zeros(3)         # low-passed GPS-velocity derivative (true inertial accel)
        self._v_gps_prev = None

    # -- convenience ---------------------------------------------------------------------
    @property
    def R(self):
        return q2R(self.q)

    @property
    def pos(self):
        return self.x[:, 0]

    @property
    def vel(self):
        return self.x[:, 1]

    @property
    def alt(self):
        return -self.x[2, 0]

    def euler(self):
        return R2euler(self.R)

    def mag_comp(self, m):
        return m - 0.8 * MOTOR_FIELD * self.thr ** 2

    # -- init ----------------------------------------------------------------------------
    def init_static(self, acc, mag, baro_alt):
        up = acc / np.linalg.norm(acc)
        m = self.mag_comp(mag)
        # build R from up (body) and magnetic north: body x,y,z in NED
        zb = -up                                   # body down axis expressed in body frame -> solve in NED
        # NED down in body = -up; east = down x mag_north_dir
        down = -up
        east = np.cross(down, m)
        east /= np.linalg.norm(east)
        north = np.cross(east, down)
        R = np.vstack([north, east, down])          # rows = NED axes in body coords = body->NED matrix
        # magnetic north sits at true heading DECL
        c, s = np.cos(DECL), np.sin(DECL)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.q = R2q(Rz @ R)
        self.alt_baro0 = baro_alt

    # -- propagation ---------------------------------------------------------------------
    def update_imu(self, gyro, acc, mag, dt, thr):
        """mag may be None. Runs at IMU rate."""
        self.thr = thr
        R = self.R
        up_b = R.T @ np.array([0.0, 0.0, -1.0])
        # remove inertial acceleration (thrust, centripetal) using GPS: f - R^T a = g * up
        m_up = acc - R.T @ self.a_gps if self.gps_ok else acc
        fn = np.linalg.norm(m_up)
        gate = max(0.0, 1.0 - abs(fn / G - 1.0) / 0.25)
        e = np.zeros(3)
        if fn > 1e-3 and gate > 0:
            e = np.cross(m_up / fn, up_b) * gate
        if mag is not None:
            m = self.mag_comp(mag)
            mn = np.linalg.norm(m)
            if mn > 1e-3:
                # use the true-north reference field (declination is known)
                m_pred = R.T @ self.m_ref
                em = np.cross(m / mn, m_pred)
                e = e + self.Kp_mag / self.Kp_acc * np.dot(em, up_b) * up_b
        if gate > 0.5:
            self.eint += e * dt
            self.eint = np.clip(self.eint, -0.5, 0.5)
        w = gyro + self.Kp_acc * e + self.Ki * self.eint
        self.q = self.q + 0.5 * qmul(self.q, np.array([0.0, *w])) * dt
        self.q /= np.linalg.norm(self.q)
        self.w = gyro + self.Ki * self.eint     # bias-corrected rates for the rate loop
        self.acc = acc

        # navigation prediction
        a_n = self.R @ acc + np.array([0, 0, G])
        self.a_ned_f += 0.2 * (a_n - self.a_ned_f)
        F = np.array([[1, dt, -0.5 * dt * dt], [0, 1, -dt], [0, 0, 1]])
        Q = np.diag([0.0, (0.35 * dt) ** 2, (0.01 * dt) ** 2])
        for i in range(3):
            u = a_n[i]
            self.x[i] = F @ self.x[i] + np.array([0.5 * dt * dt * u, dt * u, 0.0])
            self.P[i] = F @ self.P[i] @ F.T + Q

    def _update(self, i, H, z, Rm):
        P = self.P[i]
        S = H @ P @ H.T + Rm
        K = P @ H.T @ np.linalg.inv(S)
        self.x[i] = self.x[i] + K @ (z - H @ self.x[i])
        self.P[i] = (np.eye(3) - K @ H) @ P

    def update_gps(self, p, v, latency=0.10):
        if self._v_gps_prev is not None:
            a_raw = (v - self._v_gps_prev) / 0.1
            self.a_gps += 0.25 * (a_raw - self.a_gps)
        self._v_gps_prev = v.copy()
        p = p + v * latency                       # latency compensation
        v = v + self.a_ned_f * latency
        H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        for i in range(3):
            sp = 1.8 if i < 2 else 3.0
            sv = 0.25 if i < 2 else 0.4
            self._update(i, H, np.array([p[i], v[i]]), np.diag([sp * sp, sv * sv]))
        self.gps_ok = True

    def update_baro(self, alt):
        H = np.array([[1.0, 0, 0]])
        self._update(2, H, np.array([-(alt - self.alt_baro0)]), np.array([[0.6 ** 2]]))
