"""Physics sanity checks:  .venv/bin/python tests/physics_check.py

1. vacuum tumble      no air: energy and inertial angular momentum must be conserved (rigid body,
                      gyroscopic rotor, Ixz, intermediate-axis flip)
2. power audit        in still air, d(E_mech)/dt must equal the power of every external force
                      (aero, prop, motor shaft, ground) -- motor off glide and powered climb
3. flight behaviour   static thrust, trimmed glide, level speed, phugoid vs Lanchester, roll mode,
                      parked in a crosswind on the brakes
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rcsim.airframe import Airframe, G                      # noqa: E402
from rcsim.mathutil import euler2R, R2q, q2R, R2euler        # noqa: E402

DT = 0.005
ok_all = True


def check(name, cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


def state(af, alt, V, pitch=0.0, yaw=0.0, roll=0.0, w=(0, 0, 0), Om=0.0):
    x = af.initial_state((0, 0, -alt), R2q(euler2R(yaw, pitch, roll)))
    x[3:6] = q2R(x[6:10]) @ np.array([V, 0, 0])
    x[10:13] = w
    x[13] = Om
    return x


# ------------------------------------------------------------------ 1. vacuum
af = Airframe()
af.I0, af.B_visc = 0.0, 0.0
rho = 1e-9
x = state(af, 3000, 10.0, w=(0.2, 3.0, 0.1), Om=800.0)
E0 = af.energy(x, rho=rho)['total']
Hn0 = q2R(x[6:10]) @ (af.I @ x[10:13] + af.Jr * (x[10] + x[13]) * np.array([1, 0, 0]))
wy_sign_flips, last = 0, np.sign(x[11])
for _ in range(int(20 / DT)):
    x, _, _ = af.step(x, np.zeros(5), np.zeros(3), DT, rho)
    if np.sign(x[11]) != last:
        wy_sign_flips += 1
        last = np.sign(x[11])
E1 = af.energy(x, rho=rho)['total']
Hn1 = q2R(x[6:10]) @ (af.I @ x[10:13] + af.Jr * (x[10] + x[13]) * np.array([1, 0, 0]))
check('vacuum energy', abs(E1 - E0) < 1e-4 * abs(E0), f"E0 {E0:.4f} J  E20s {E1:.4f} J")
check('vacuum angular momentum', np.linalg.norm(Hn1 - Hn0) < 1e-4 * np.linalg.norm(Hn0),
      f"|H| {np.linalg.norm(Hn0):.5f} drift {np.linalg.norm(Hn1 - Hn0):.2e}")
check('intermediate-axis (pitch) spin is unstable', wy_sign_flips >= 2, f"{wy_sign_flips} pitch-rate reversals in 20 s")


# ------------------------------------------------------------------ 2. power audit
def audit(duty, V, pitch, secs, label):
    af = Airframe()
    x = state(af, 1500, V, pitch=pitch, Om=900.0 if duty > 0 else 300.0)
    u = np.array([0.0, 0.0, 0.0, duty, 0.0])
    E0 = af.energy(x)['total']
    work, P_prev = 0.0, None
    for _ in range(int(secs / DT)):
        af.deriv(x, u, np.zeros(3))
        P0 = af.info['P_ext']
        x, _, _ = af.step(x, u, np.zeros(3), DT)
        af.deriv(x, u, np.zeros(3))
        work += 0.5 * (P0 + af.info['P_ext']) * DT
    E1 = af.energy(x)['total']
    err = abs((E1 - E0) - work)
    check(f'power audit ({label})', err < 0.01 * max(abs(work), 1.0),
          f"dE {E1 - E0:8.2f} J   work of forces {work:8.2f} J   mismatch {err:.3f} J")


audit(0.0, 14.0, 0.0, 8.0, 'motor off')
audit(0.75, 13.0, 0.2, 8.0, 'powered')

# ------------------------------------------------------------------ 3. behaviour
af = Airframe()
print('\nmass properties:', af.summary())

# static thrust
x = state(af, 0.1, 0.0)
for _ in range(int(2 / DT)):
    x[13] = af.step(x, np.array([0, 0, 0, 1.0, 1.0]), np.zeros(3), DT)[0][13]
xd, _ = af.deriv(x, np.array([0, 0, 0, 1.0, 1.0]), np.zeros(3))
i = af.info
check('static thrust', 1.8 < i['T'] / (af.m * G) < 3.0,
      f"T {i['T']:.1f} N = {i['T'] / (af.m * G):.2f} x weight, {i['rpm']:.0f} rpm, motor {i['I']:.1f} A, "
      f"battery {i['Ib']:.1f} A @ {i['Vb']:.1f} V")


def trim_glide(af, de=0.12, secs=50.0):
    """Hands-off glide (wings held level) with a fixed elevator trim; let the phugoid die out and
    measure speed / sink / glide ratio over the last 10 s."""
    x = state(af, 900, 12.0, pitch=-0.05)
    u = np.array([0.0, de, 0.0, 0.0, 0.0])
    for k in range(int(secs / DT)):
        x, _, _ = af.step(x, u, np.zeros(3), DT)
        x[10] = x[12] = 0.0
        if k == int((secs - 10) / DT):
            x0 = x.copy()
    dist = np.hypot(*(x[0:2] - x0[0:2]))
    drop = x[2] - x0[2]
    return de, np.linalg.norm(x[3:6]), drop / 10.0, dist / drop, x


de, V, sink, LD, xg = trim_glide(af)
check('glide ratio (flat-plate 3D foamie ~4-8)', 3.5 < LD < 8.5,
      f"elevator {de:+.2f}: {V:.1f} m/s, sink {sink:.2f} m/s, L/D {LD:.1f}, windmilling prop {xg[13] * 9.549:.0f} rpm")

# phugoid: kick the trimmed glide and measure the speed oscillation period
x = xg.copy()
x[3:6] *= 1.15
ts, vs = [], []
for k in range(int(30 / DT)):
    x, _, _ = af.step(x, np.array([0, de, 0, 0, 0]), np.zeros(3), DT)
    x[10] = x[12] = 0
    if k % 4 == 0:
        ts.append(k * DT)
        vs.append(np.linalg.norm(x[3:6]))
vs = np.array(vs) - np.mean(vs)
zc = [ts[k] for k in range(1, len(vs)) if vs[k - 1] < 0 <= vs[k]]
Tph = np.mean(np.diff(zc)) if len(zc) > 2 else float('nan')
Tlan = np.pi * np.sqrt(2) * V / G
check('phugoid period ~ Lanchester pi*sqrt(2)*V/g', 0.8 * Tlan < Tph < 1.6 * Tlan,
      f"{Tph:.2f} s vs {Tlan:.2f} s (Lanchester assumes constant CL; pitch damping q*l/V is not small at "
      f"model scale, so CL dips at the fast end and the period stretches)")

# roll mode: step aileron at 13 m/s, time to 63 % of steady roll rate
x = xg.copy()
p = []
for k in range(int(1.5 / DT)):
    x, _, _ = af.step(x, np.array([0.5, de, 0, 0, 0]), np.zeros(3), DT)
    p.append(x[10])
p = np.array(p)
pss = p[int(0.8 / DT):].max()
tau_r = np.argmax(p > 0.63 * pss) * DT
check('roll subsidence time constant (small RC 0.05-0.25 s)', 0.04 < tau_r < 0.25,
      f"tau {tau_r * 1000:.0f} ms, steady roll rate {np.degrees(pss):.0f} deg/s at half aileron")

# level flight: full throttle top speed
x = state(af, 100, 15.0, Om=1000)
for _ in range(int(25 / DT)):
    x, _, _ = af.step(x, np.array([0, 0, 0, 1.0, 0]), np.zeros(3), DT)
    R = q2R(x[6:10])
    yaw, pit, rol = R2euler(R)
    # crude level-flight hold so the steady speed is meaningful
    x[10:13] = 0
    x[6:10] = R2q(euler2R(0, float(np.clip(pit, -0.2, 0.2)), 0))
    x[5] = 0
V_top = np.linalg.norm(x[3:6])
check('top speed at full throttle (10x5 prop, pitch speed ~25 m/s)', 18 < V_top < 32, f"{V_top:.1f} m/s")

# zoom climb: kinetic energy turns into height (minus drag work), the aeroplane does not just stop
x = state(af, 100, 22.0)
KE0 = 0.5 * af.m * 22.0 ** 2
h0 = -x[2]
hmax = h0
for _ in range(int(6 / DT)):
    x, _, _ = af.step(x, np.array([0, 0.45, 0, 0, 0]), np.zeros(3), DT)
    x[10] = x[12] = 0.0
    if -x[2] > hmax:
        hmax, Vtop = -x[2], np.linalg.norm(x[3:6])
gain = af.m * G * (hmax - h0) / (KE0 - 0.5 * af.m * Vtop ** 2)
check('zoom climb: KE -> PE', 0.5 < gain < 1.0,
      f"from 22 m/s, motor off: +{hmax - h0:.1f} m, {Vtop:.1f} m/s over the top, "
      f"{gain * 100:.0f} % of the kinetic energy given up became height (rest = drag)")

# parked on the brakes in a crosswind: physical tyre/brake friction holds it (no teleporting)
for Vw, must_hold in ((4.0, True), (7.0, False)):
    af2 = Airframe()
    x = af2.initial_state((0, 0, -0.10))
    for _ in range(int(2 / DT)):
        x, _, _ = af2.step(x, np.array([0, 0, 0, 0, 1.0]), np.zeros(3), DT)
    for k in range(int(10 / DT)):
        x, _, _ = af2.step(x, np.array([0, 0, 0, 0, 1.0]), np.array([0, -Vw, 0]), DT)
        if k == int(5 / DT):
            p5, y5 = x[0:2].copy(), R2euler(q2R(x[6:10]))[0]
    creep = np.linalg.norm(x[0:2] - p5)
    dyaw = np.degrees(R2euler(q2R(x[6:10]))[0])
    msg = f"{Vw:.0f} m/s straight across: settles at yaw {dyaw:+.1f} deg, creep in last 5 s {creep * 1000:.1f} mm"
    if must_hold:
        check('parked on brakes, 4 m/s crosswind', creep < 0.002 and abs(dyaw) < 5, msg)
    else:
        print(f"[INFO] parked on brakes, {msg} (a light foamie weathervanes in a strong crosswind)")

print('\nALL PASS' if ok_all else '\nSOME CHECKS FAILED')
