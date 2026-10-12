"""Sensor models: MPU-class IMU, QMC5883L compass, BMP280 baro, u-blox M10 GPS (SEQURE M10-25Q).

Noise levels are meant to be realistic *in an airframe* (vibration, correlated GPS error,
motor-current magnetic interference), not datasheet-lab numbers.
"""
import numpy as np
from .mathutil import q2R

DECL = np.radians(-13.0)   # NYC
INCL = np.radians(67.0)
B_TOTAL = 51.5            # uT
MAG_NED = B_TOTAL * np.array([np.cos(INCL) * np.cos(DECL), np.cos(INCL) * np.sin(DECL), np.sin(INCL)])
MOTOR_FIELD = np.array([0.3, 0.5, -0.8]) * 2.5   # uT at full throttle^2 (wire/ESC current)


class GaussMarkov:
    def __init__(self, rng, sigma, tau, n):
        self.rng, self.sigma, self.tau = rng, np.asarray(sigma, float), tau
        self.x = rng.normal(0, 1, n) * self.sigma

    def step(self, dt):
        a = np.exp(-dt / self.tau)
        self.x = a * self.x + self.sigma * np.sqrt(1 - a * a) * self.rng.normal(0, 1, self.x.shape)
        return self.x


class Sensors:
    def __init__(self, seed=1, dt=0.005):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.dt = dt
        # IMU
        self.gyro_bias = r.normal(0, 0.012, 3)            # rad/s turn-on bias
        self.gyro_sigma = 0.012                            # rad/s incl. vibration
        self.acc_bias = r.normal(0, 0.12, 3)               # m/s^2
        self.acc_sigma = 0.20
        self.gyro_rw = 0.0004
        self.gyro_scale = 1 + r.normal(0, 0.004, 3)
        self.acc_scale = 1 + r.normal(0, 0.005, 3)
        e = r.normal(0, 0.004, 3)                          # ~0.25 deg axis misalignment
        self.acc_mis = np.eye(3) + np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        self.soft_iron = np.eye(3) + r.normal(0, 0.015, (3, 3))
        # mag (QMC5883L): residual hard-iron after cal, noise
        self.hard_iron = r.normal(0, 0.8, 3)
        self.mag_sigma = 0.25
        # baro
        self.baro_bias = GaussMarkov(r, 0.8, 300.0, 1)
        self.baro_sigma = 0.15
        # GPS
        self.gps_pos_gm = GaussMarkov(r, [1.0, 1.0, 1.8], 40.0, 3)
        self.gps_vel_gm = GaussMarkov(r, [0.03, 0.03, 0.06], 5.0, 3)
        self.gps_latency = 0.10
        self.gps_buf = []
        self.t = 0.0

    def imu(self, fspec, w, phase=0.0, rpm_f=0.0):
        """phase = rotor angle, rpm_f = rotor speed / full speed. Prop vibration is sampled at the IMU
        rate so it aliases exactly like a real unfiltered 200 Hz gyro/accel does."""
        self.gyro_bias += self.rng.normal(0, self.gyro_rw * np.sqrt(self.dt), 3)
        vib = rpm_f * rpm_f
        gyro = self.gyro_scale * w + self.gyro_bias + self.rng.normal(0, self.gyro_sigma * (1 + 0.8 * vib), 3)
        gyro = gyro + vib * 0.02 * np.array([np.sin(phase), np.cos(phase * 1.01), np.sin(phase * 0.99 + 1)])
        a = self.acc_mis @ (self.acc_scale * fspec)
        acc = a + self.acc_bias + self.rng.normal(0, self.acc_sigma * (1 + 1.0 * vib), 3)
        acc = acc + vib * 0.35 * np.array([np.sin(phase + 2), np.sin(phase * 1.02), np.cos(phase)])
        return gyro, acc

    def mag(self, R, thr):
        m = self.soft_iron @ (R.T @ MAG_NED) + self.hard_iron + MOTOR_FIELD * thr * thr
        return m + self.rng.normal(0, self.mag_sigma, 3)

    def baro(self, alt):
        self.baro_bias.step(1 / 25)
        return alt + self.baro_bias.x[0] + self.rng.normal(0, self.baro_sigma)

    def gps(self, pos, vel, t):
        """Call at 10 Hz. Returns a (pos, vel) fix delayed by latency, or None."""
        self.gps_pos_gm.step(0.1)
        self.gps_vel_gm.step(0.1)
        p = pos + self.gps_pos_gm.x + self.rng.normal(0, [0.25, 0.25, 0.4])
        v = vel + self.gps_vel_gm.x + self.rng.normal(0, [0.05, 0.05, 0.10])
        self.gps_buf.append((t + self.gps_latency, p, v))
        if self.gps_buf and self.gps_buf[0][0] <= t:
            _, p, v = self.gps_buf.pop(0)
            return p, v
        return None
