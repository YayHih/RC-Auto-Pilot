"""Glue: physics (200 Hz) + atmosphere + sensor chips + the C++ flight controller (100 Hz, the same
code that runs on the Heltec) + PCA9685 + servos.

The flight controller only ever sees raw register values and a microsecond clock, and only ever
produces PCA9685 counts; see rcsim/hw.py and firmware/RCAutopilot/src/fc.
"""
import numpy as np
from .airframe import Airframe, isa_rho
from .sensors import Sensors
from .mathutil import euler2R, R2q, q2R, R2euler
from . import fcbridge as fb
from .hw import MPU6050, PCA9685, mag_counts, baro_q8, nav_pvt

DT = 0.005
FT = 3.28084
G_ = 9.80665


class Atmosphere:
    """Log-profile mean wind + Dryden turbulence (MIL-F-8785C low-altitude form), speed-scaled,
    plus slow wind speed/direction wander and discrete 1-cos gust events.

    Everything can be changed while running (set_wind): speed in m/s at 6 m, direction the wind blows
    FROM in degrees, turbulence intensity multiplier, gust frequency multiplier."""

    def __init__(self, rng, wind_speed=2.5, wind_from_deg=250.0, turb=1.0, gusts=1.0, wander=1.0):
        self.rng = rng
        self.W20 = wind_speed                       # mean wind at 6 m
        self.from_deg = wind_from_deg
        self.turb = turb
        self.gusts = gusts
        self.wander = wander
        self.g = np.zeros(3)
        self.z0 = 0.05
        self.w_mul = 0.0                            # slow speed wander (fraction)
        self.d_off = 0.0                            # slow direction wander (rad)
        self.events = []                            # [t0, dur, amp-vector]
        self.t = 0.0

    @property
    def dir(self):
        """direction the wind blows towards (rad, NED azimuth), including wander"""
        return np.radians(self.from_deg + 180.0) + self.d_off

    def set_wind(self, speed=None, from_deg=None, turb=None, gusts=None):
        if speed is not None:
            self.W20 = max(0.0, float(speed))
        if from_deg is not None:
            self.from_deg = float(from_deg) % 360.0
        if turb is not None:
            self.turb = max(0.0, float(turb))
        if gusts is not None:
            self.gusts = max(0.0, float(gusts))

    def mean(self, h):
        h = max(h, 0.3)
        s = self.W20 * (1.0 + self.w_mul) * np.log(h / self.z0) / np.log(6.0 / self.z0)
        d = self.dir
        return np.array([s * np.cos(d), s * np.sin(d), 0.0])

    def _gust_events(self, dt):
        rate = self.gusts * (0.03 + 0.05 * min(self.W20, 6.0) / 6.0) * max(self.turb, 0.0)
        if self.rng.random() < rate * dt:
            amp = self.rng.uniform(0.6, 2.0) * (1.0 + 0.35 * self.W20) * 0.8
            az = self.rng.uniform(0, 2 * np.pi)
            v = np.array([np.cos(az), np.sin(az), self.rng.normal(0, 0.4)]) * amp
            self.events.append([self.t, self.rng.uniform(1.2, 4.0), v])
        out = np.zeros(3)
        keep = []
        for ev in self.events:
            x = (self.t - ev[0]) / ev[1]
            if x < 1.0:
                out += ev[2] * 0.5 * (1.0 - np.cos(2 * np.pi * x))
                keep.append(ev)
        self.events = keep
        return out

    def step(self, dt, alt, V):
        self.t += dt
        # slow Gauss-Markov wander of the mean wind
        a = np.exp(-dt / 45.0)
        self.w_mul = a * self.w_mul + self.wander * 0.18 * np.sqrt(1 - a * a) * self.rng.normal()
        self.d_off = a * self.d_off + self.wander * np.radians(10.0) * np.sqrt(1 - a * a) * self.rng.normal()
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
        if self.gusts > 0.0 and alt > 0.5:
            w = w + self._gust_events(dt)
        return w


class Servo:
    """Hobby servo: latches a pulse each 20 ms frame, deadband, backlash, rate limit, first-order lag."""

    def __init__(self, rng, rate=14.0, tau=0.025):
        self.pos, self.rate, self.tau = 0.0, rate, tau
        self.hold = 0.0
        self.back = 0.004
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


class _APView:
    """Read-only view of the flight controller's autopilot telemetry (what the viewer shows)."""

    def __init__(self, sim):
        self._s = sim

    @property
    def state(self):
        return fb.STATE_NAMES[self._s.fout.state]

    @property
    def goal(self):
        return fb.CMD_NAMES.get(self._s.fout.goal)


class _EstView:
    def __init__(self, sim):
        self._s = sim

    @property
    def pos(self):
        return np.array(self._s.fout.pos[:])

    @property
    def vel(self):
        return np.array(self._s.fout.vel[:])

    @property
    def alt(self):
        return -float(self._s.fout.pos[2])

    @property
    def q(self):
        return np.array(self._s.fout.q[:])

    @property
    def R(self):
        return q2R(self.q)

    def euler(self):
        return R2euler(self.R)

    @property
    def wind(self):
        return np.array(self._s.fout.wind[:])


class Sim:
    """seed fixes every random draw (sensors, build tolerances, wind, turbulence, gusts) so a run can be
    replayed.  wind / wind_from default to a random draw from the seed; randomize scales the airframe
    tolerances (0 = nominal model); plane = dict of factors for a different aeroplane
    (ail, ele, rud, mass, thrust - see Airframe.modify).

    fc = 'native'   C++ flight controller compiled for this PC (fast; default)
         'arm'      the Cortex-M0+ machine code in the Unicorn emulator, cycle-counted at 48 MHz
         'lockstep' both, asserting bit-identical outputs every tick"""

    IO_READ_US = 450          # I2C 400 kHz: MPU6050 burst (+ compass / baro on their ticks)
    IO_WRITE_US = 520         # I2C: 5 PCA9685 channels
    NATIVE_COMPUTE_US = 2800  # typical in-flight compute on the M0+ (measured with fc='arm')

    def __init__(self, seed=1, wind=None, wind_from=None, turb=1.0, randomize=1.0, gusts=1.0,
                 fc='native', plane=None, cfg=None, cycle_every=10):
        self.af = Airframe()
        self.af.randomize(np.random.default_rng(seed + 200), randomize)
        if plane:
            self.af.modify(**plane)
        wr = np.random.default_rng(seed + 300)
        if wind is None:
            wind = float(wr.uniform(0.0, 4.5))
        if wind_from is None:
            wind_from = float(wr.uniform(0.0, 360.0))
        self.sens = Sensors(seed, DT)
        self.rng = np.random.default_rng(seed + 100)
        self.atm = Atmosphere(self.rng, wind, wind_from, turb, gusts)
        if fc == 'arm':
            self.fc = fb.ArmFC(cfg, cycle_every)
        elif fc == 'lockstep':
            self.fc = fb.LockstepFC(cfg, cycle_every)
        else:
            self.fc = fb.NativeFC(cfg)
        self.fc_kind = fc
        self.cfg = self.fc.cfg
        self.pca = PCA9685(self.cfg.pca_osc_hz, self.cfg.pca_prescale, float(self.rng.uniform()))
        self.mpu = MPU6050(DT)
        self.fin = fb.FcIn()
        self.fout = fb.FcOut()
        self.ap = _APView(self)
        self.est = _EstView(self)
        self.servos = [Servo(self.rng) for _ in range(4)]          # ail, ele, rud, brake
        self.esc_cmd = 0.0
        self.x = np.zeros(self.af.NX)
        self.x[2] = -0.10
        self.x[6:10] = R2q(euler2R(0.0, 0.0, 0.0))
        self.servo = np.zeros(3)
        self.brake = 1.0
        self.duty = 0.0
        self.mot = 0.0
        self.t = 0.0
        self.clock = 0.0                 # never reset: sensor / FC / PCA time base
        self.n = 0
        self.fspec = np.array([0, 0, -9.80665])
        self.T = 0.0
        self.rotor_phase = 0.0
        self.wind_now = np.zeros(3)
        self.log = []
        self.cycle_log = []
        self._cmd = 0
        # manual flying: stick/throttle inputs in -1..1 / 0..1, 'assist' = stabilised angle mode
        self.manual = False
        self.assist = True
        self.stick = dict(da=0.0, de=0.0, dr=0.0, thr=0.0)
        self._settle()

    @property
    def info(self):
        return self.af.info

    @property
    def out_now(self):
        c = self.fout.cmd
        return dict(da=c[0], de=c[1], dr=c[2], thr=c[3], brake=c[4])

    def _settle(self):
        """Power-up on the runway: the aeroplane settles on its gear while the flight controller boots,
        levels its attitude filter and takes its GPS home."""
        for _ in range(int(2.0 / DT)):
            self._step_core()
        for _ in range(int(15.0 / DT)):
            self._step_core()
            if self.fout.state == 1 and (self.fout.flags & 2):     # GROUND with GPS
                break
        self.t = 0.0
        self.n = 0
        self.log = []

    # ------------------------------------------------------------------ flight controller I/O
    def _fc_tick(self):
        k = int(round(self.clock / DT)) // 2                 # FC tick number (100 Hz)
        fi = self.fin
        fi.t_us = int(round(self.clock * 1e6)) & 0xFFFFFFFF
        flags = fb.IN_IMU
        a, g = self.mpu.read()
        fi.acc[:] = a.tolist()
        fi.gyro[:] = g.tolist()
        R = q2R(self.x[6:10])
        if k % 2 == 0:
            cur = min(1.0, self.af.info['I'] / 25.0) ** 0.5   # motor-wire field is linear in current
            fi.mag[:] = mag_counts(self.sens.mag(R, cur)).tolist()
            flags |= fb.IN_MAG
        if k % 4 == 0:
            fi.baro_q8 = baro_q8(self.sens.baro(-self.x[2]))
            flags |= fb.IN_BARO
        if k % 10 == 0:
            fix = self.sens.gps(self.x[0:3], self.x[3:6], self.clock)
            if fix is not None:
                lat, lon, h, v = nav_pvt(*fix)
                fi.lat_e7, fi.lon_e7, fi.hmsl_mm = lat, lon, h
                fi.vel_mm_s[:] = v
                fi.fix_type, fi.num_sv, fi.hacc_mm = 3, 12, 1500
                flags |= fb.IN_GPS
        fi.flags = flags
        fi.cmd = self._cmd
        self._cmd = 0
        if self.manual:
            st = self.stick
            fi.rc_mode = fb.RC_ASSIST if self.assist else fb.RC_DIRECT
            fi.rc_us[:] = [int(1500 + 500 * np.clip(st['da'], -1, 1)), int(1500 + 500 * np.clip(st['de'], -1, 1)),
                           int(1000 + 1000 * np.clip(st['thr'], 0, 1)), int(1500 + 500 * np.clip(st['dr'], -1, 1))]
        else:
            fi.rc_mode = fb.RC_AUTO
            fi.rc_us[:] = [1500, 1500, 1000, 1500]
        self.fout = self.fc.tick(fi)
        cyc = getattr(self.fc, 'cycles', None)
        compute = cyc / 48e6 if cyc else self.NATIVE_COMPUTE_US * 1e-6
        if cyc and self.fc.cycles is not None and (k % max(getattr(self.fc, 'cycle_every', 1) or 1, 1) == 0):
            self.cycle_log.append((self.t, fb.STATE_NAMES[self.fout.state], cyc))
        lat = (self.IO_READ_US + self.IO_WRITE_US) * 1e-6 + compute
        self.pca.write(self.clock + lat, list(self.fout.pca_off[:5]))

    def _pulse_to_cmd(self, ch, us):
        c = self.cfg
        return (us - c.ch_center_us[ch]) / c.ch_throw_us[ch] * c.ch_reverse[ch]

    # ------------------------------------------------------------------ one 5 ms physics step
    def _step_core(self):
        if self.n % 2 == 0:
            self._fc_tick()
        for _, pulses in self.pca.frames_until(self.clock + DT):
            for i in range(4):
                ch = i if i < 3 else 4
                if pulses[ch] is not None:
                    self.servos[i].command(float(np.clip(self._pulse_to_cmd(ch, pulses[ch]), -1.2, 1.2)))
            p = pulses[3]
            self.esc_cmd = 0.0 if p is None else float(np.clip((self._pulse_to_cmd(3, p) + 1.0) / 2.0, 0.0, 1.0))
        for i in range(3):
            self.servo[i] = self.servos[i].step(DT)
        self.brake = float(np.clip((self.servos[3].step(DT) + 1.0) / 2.0, 0.0, 1.0))
        # ESC slew limit
        d = np.clip(self.esc_cmd - self.duty, -12 * DT, 12 * DT)
        self.duty = float(np.clip(self.duty + d, 0, 1))
        self.mot = self.duty
        alt = -self.x[2]
        V = float(np.linalg.norm(self.x[3:6]))
        self.wind_now = self.atm.step(DT, alt, V)
        rho = isa_rho(alt)
        u = np.array([*self.servo, self.duty, self.brake])
        self.x, self.fspec, self.T = self.af.step(self.x, u, self.wind_now, DT, rho)
        self.rotor_phase = (self.rotor_phase + self.x[13] * DT) % (2 * np.pi)
        rpm_f = self.x[13] / 1200.0
        gyro, acc = self.sens.imu(self.fspec, self.x[10:13], self.rotor_phase, rpm_f)
        self.mpu.sample(gyro, acc)
        self.clock += DT
        self.n += 1

    def set_wind(self, speed=None, from_deg=None, turb=None, gusts=None):
        self.atm.set_wind(speed, from_deg, turb, gusts)

    def set_manual(self, on):
        """Flip the RC mode switch: take over with the stick, or hand back to the autopilot (which joins
        its loiter circle if airborne)."""
        on = bool(on)
        if on == self.manual:
            return
        self.manual = on
        if on:
            self.stick.update(da=0.0, de=0.0, dr=0.0, thr=float(self.duty))

    def step(self):
        self._step_core()
        self.t += DT
        if self.n % 10 == 0:
            self._log()

    def _log(self):
        R = q2R(self.x[6:10])
        ey, ep, er = R2euler(R)
        o = self.fout
        self.log.append(dict(
            t=self.t, state=fb.STATE_NAMES[o.state], pos=self.x[0:3].copy(), vel=self.x[3:6].copy(),
            eul=np.array([ey, ep, er]), eeul=np.array(self.est.euler()), epos=np.array(o.pos[:]),
            evel=np.array(o.vel[:]), thr=self.duty, surf=self.servo.copy(), T=self.T,
            crashed=self.af.crashed, R=R, wind_est=np.array(o.wind[:]), wind_true=self.wind_now.copy(),
            airspeed_est=o.airspeed, airspeed=float(np.linalg.norm(self.x[3:6] - self.wind_now)),
            auth_q=np.array(o.auth_q[:]), auth_w=np.array(o.auth_w[:]), pca=list(o.pca_off[:5])))

    def press(self, name):
        self._cmd = fb.CMDS.get(name, 0)

    def run(self, seconds):
        for _ in range(int(round(seconds / DT))):
            self.step()
            if self.af.crashed:
                break
