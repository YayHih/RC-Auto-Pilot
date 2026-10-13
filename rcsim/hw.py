"""Hardware emulation between the physics and the flight controller: the exact numbers each chip
puts in its registers, and what the servos make of the PCA9685 pulse train.

  MPU6050   ACCEL/GYRO_xOUT: internal 1 kHz sampling through the 42/44 Hz DLPF (DLPF_CFG = 3),
            +-8 g = 4096 LSB/g, +-1000 dps = 32.8 LSB/(deg/s), int16 saturation
  QMC5883L  X/Y/Z at 50 Hz ODR, +-8 G = 3000 LSB/G
  BMP280    compensated pressure as the Bosch driver returns it: Pa * 256 (uint32)
  u-blox    UBX-NAV-PVT: lat/lon 1e-7 deg, hMSL mm, velN/E/D mm/s
  PCA9685   free-running 50 Hz frames from its own oscillator; each servo / ESC latches whatever OFF
            count is in the register when its frame starts
"""
import numpy as np
from .airframe import isa_rho  # noqa: F401  (kept for callers that want air density)

G = 9.80665
HOME_LAT, HOME_LON, HOME_HMSL = 40.7128, -74.0060, 10.0
M_PER_DEG = 6371000.0 * np.pi / 180.0


def _i16(x):
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


class MPU6050:
    def __init__(self, dt):
        # DLPF ~42 Hz as two first-order sections on the physics-rate stream
        tau = 1.0 / (2 * np.pi * 42.0 * 1.55)
        self.a = dt / (tau + dt)
        self.f1 = np.zeros(6)
        self.f2 = np.zeros(6)
        self.first = True

    def sample(self, gyro, acc):
        x = np.concatenate([gyro, acc])
        if self.first:
            self.f1[:] = x
            self.f2[:] = x
            self.first = False
        self.f1 += self.a * (x - self.f1)
        self.f2 += self.a * (self.f1 - self.f2)

    def read(self):
        g = _i16(self.f2[:3] * (180.0 / np.pi) * 32.8)
        a = _i16(self.f2[3:] / G * 4096.0)
        return a, g


def mag_counts(b_uT):
    return _i16(np.asarray(b_uT) * 30.0)


def baro_q8(alt_m):
    p = 101325.0 * (1.0 - 2.25577e-5 * alt_m) ** 5.25588
    return int(round(p * 256.0)) & 0xFFFFFFFF


def nav_pvt(p_ned, v_ned):
    lat = HOME_LAT + p_ned[0] / M_PER_DEG
    lon = HOME_LON + p_ned[1] / (M_PER_DEG * np.cos(np.radians(HOME_LAT)))
    return (int(round(lat * 1e7)), int(round(lon * 1e7)), int(round((HOME_HMSL - p_ned[2]) * 1000.0)),
            [int(round(v * 1000.0)) for v in v_ned])


class PCA9685:
    """Output side: tracks register writes in time and serves them to the servos at frame starts."""

    def __init__(self, osc_hz, prescale, phase, nch=5):
        self.period = 4096.0 * (prescale + 1) / osc_hz
        self.us_per_count = (prescale + 1) * 1e6 / osc_hz
        self.next_frame = phase * self.period
        self.pending = []                    # (time effective, counts tuple)
        self.reg = [0] * nch                 # outputs off until the first write
        self.nch = nch

    def write(self, t_eff, counts):
        self.pending.append((t_eff, tuple(int(c) for c in counts[:self.nch])))

    def frames_until(self, t):
        """Yield (frame time, pulse widths us) for every frame starting at or before t."""
        out = []
        while self.next_frame <= t:
            tf = self.next_frame
            while self.pending and self.pending[0][0] <= tf:
                self.reg = list(self.pending.pop(0)[1])
            out.append((tf, [c * self.us_per_count if c > 0 else None for c in self.reg]))
            self.next_frame += self.period
        return out
