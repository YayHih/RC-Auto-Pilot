"""Glue: physics + servos + atmosphere + sensors + estimator + autopilot, stepped at 200 Hz."""
import numpy as np
from .airframe import Airframe, isa_rho
from .sensors import Sensors
from .estimator import Estimator
from .autopilot import Autopilot
from .mathutil import euler2R, R2q, q2R, R2euler

DT = 0.005
FT = 3.28084


class Atmosphere:
    """Log-profile mean wind + Dryden turbulence (MIL-F-8785C low-altitude form), speed-scaled."""

    def __init__(self, rng, wind_speed=2.5, wind_from_deg=250.0, turb=1.0):
        self.rng = rng
        self.W20 = wind_speed                       # mean wind at 6 m
        self.dir = np.radians(wind_from_deg + 180.0)   # direction the wind blows towards (from-> to)
        self.turb = turb
        self.g = np.zeros(3)
        self.z0 = 0.05

    def mean(self, h):
        h = max(h, 0.3)
        s = self.W20 * np.log(h / self.z0) / np.log(6.0 / self.z0)
        return np.array([s * np.cos(self.dir), s * np.sin(self.dir), 0.0])

    def step(self, dt, alt, V):
        h = max(alt, 2.0)
        hf = h * FT
        k = 0.177 + 0.000823 * hf
        Lw = max(h, 3.0)
        Lu = max(hf / k ** 1.2 / FT, 3.0)
        sw = 0.1 * self.W20 * self.turb + 0.05 * self.turb
        su = sw / k ** 0.4
        Vs = max(V, 4.0)
        for i, (L, s) in enumerate(((Lu, su), (Lu, su), (Lw, sw))):
            a = np.exp(-dt * Vs / L)
            self.g[i] = a * self.g[i] + s * np.sqrt(1 - a * a) * self.rng.normal()
        w = self.mean(alt) + self.g
        return w


class Servo:
    """Hobby servo: 50 Hz command, deadband, backlash, rate limit, first-order lag."""

    def __init__(self, rng, rate=14.0, tau=0.025):
        self.pos, self.tgt, self.rate, self.tau = 0.0, 0.0, rate, tau
        self.rng = rng
        self.hold = 0.0
        self.back = 0.004
        self.lastdir = 0.0
        self.out = 0.0

    def command(self, u):
        if abs(u - self.hold) > 0.004:           # deadband
            self.hold = u

    def step(self, dt):
        a = 1 - np.exp(-dt / self.tau)
        d = np.clip((self.hold - self.pos) * a, -self.rate * dt, self.rate * dt)
        self.pos += d
        # backlash: output only follows once the slack has been taken up
        if abs(self.pos - self.out) > self.back:
            self.out = self.pos - np.sign(self.pos - self.out) * self.back
        return self.out


class Sim:
    def __init__(self, seed=1, wind=2.5, wind_from=250.0, turb=1.0):
        self.af = Airframe()
        self.sens = Sensors(seed, DT)
        self.est = Estimator()
        self.ap = Autopilot()
        self.rng = np.random.default_rng(seed + 100)
        self.atm = Atmosphere(self.rng, wind, wind_from, turb)
        self.servos = [Servo(self.rng), Servo(self.rng), Servo(self.rng)]
        self.x = np.zeros(self.af.NX)
        self.x[2] = -0.10
        self.x[6:10] = R2q(euler2R(0.0, 0.0, 0.0))
        self.servo = np.zeros(3)
        self.duty = 0.0
        self.mot = 0.0
        self.t = 0.0
        self.n = 0
        self.fspec = np.array([0, 0, -9.80665])
        self.T = 0.0
        self.rotor_phase = 0.0
        self.wind_now = np.zeros(3)
        self.log = []
        self._settle()

    @property
    def info(self):
        return self.af.info

    def _settle(self):
        for _ in range(int(2.0 / DT)):
            self._physics(np.zeros(3), 0.0, 1.0)
        accs, mags, bars = [], [], []
        for _ in range(100):
            self._physics(np.zeros(3), 0.0, 1.0)
            g, a = self.sens.imu(self.fspec, self.x[10:13], 0.0, 0.0)
            accs.append(a)
            mags.append(self.sens.mag(q2R(self.x[6:10]), 0.0))
            bars.append(self.sens.baro(-self.x[2]))
        self.est.init_static(np.mean(accs, 0), np.mean(mags, 0), np.mean(bars))
        self.est.x[:, 0] = [0.0, 0.0, 0.0]
        self.t = 0.0
        self.n = 0

    def _physics(self, cmd, thr, brake):
        # 50 Hz servo frame; servo dynamics every tick
        if self.n % 4 == 0:
            for i in range(3):
                self.servos[i].command(float(np.clip(cmd[i], -1, 1)))
        for i in range(3):
            self.servo[i] = self.servos[i].step(DT)
        # ESC slew limit
        d = np.clip(thr - self.duty, -12 * DT, 12 * DT)
        self.duty = float(np.clip(self.duty + d, 0, 1))
        self.mot = self.duty
        alt = -self.x[2]
        V = float(np.linalg.norm(self.x[3:6]))
        self.wind_now = self.atm.step(DT, alt, V)
        rho = isa_rho(alt)
        u = np.array([*self.servo, self.duty, brake])
        self.x, self.fspec, self.T = self.af.step(self.x, u, self.wind_now, DT, rho)
        self.rotor_phase = (self.rotor_phase + self.x[13] * DT) % (2 * np.pi)

    def step(self):
        o = self.ap.out
        self._physics(np.array([o['da'], o['de'], o['dr']]), o['thr'], o['brake'])
        self.n += 1
        self.t += DT
        s, e = self.sens, self.est
        rpm_f = self.x[13] / 1200.0
        gyro, acc = s.imu(self.fspec, self.x[10:13], self.rotor_phase, rpm_f)
        cur = min(1.0, self.af.info['I'] / 25.0) ** 0.5       # motor-wire field is linear in current
        mag = s.mag(q2R(self.x[6:10]), cur) if self.n % 4 == 0 else None
        e.update_imu(gyro, acc, mag, DT, min(1.0, self.af.info['I'] / 25.0) ** 0.5)
        if self.n % 8 == 0:
            e.update_baro(s.baro(-self.x[2]))
        if self.n % 20 == 0:
            fix = s.gps(self.x[0:3], self.x[3:6], self.t)
            if fix is not None:
                e.update_gps(*fix)
        self.ap.update(e, DT)
        if self.n % 10 == 0:
            self._log()

    def _log(self):
        R = q2R(self.x[6:10])
        ey, ep, er = R2euler(R)
        Re = self.est.euler()
        self.log.append(dict(
            t=self.t, state=self.ap.state, pos=self.x[0:3].copy(), vel=self.x[3:6].copy(),
            eul=np.array([ey, ep, er]), eeul=np.array(Re), epos=self.est.pos.copy(),
            evel=self.est.vel.copy(), thr=self.duty, surf=self.servo.copy(),
            T=self.T, crashed=self.af.crashed, R=R))

    def press(self, name):
        self.ap.command(name)

    def run(self, seconds):
        for _ in range(int(round(seconds / DT))):
            self.step()
            if self.af.crashed:
                break
