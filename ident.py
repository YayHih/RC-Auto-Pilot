"""Measure control power (rad/s^2 per unit command) in hover and cruise, plus trim."""
import numpy as np
from rcsim.airframe import Airframe
from rcsim.mathutil import euler2R, R2q

af = Airframe()


def settle_rotor(duty, vel_b, R, t=1.5):
    x = af.initial_state()
    x[2] = -60
    x[6:10] = R2q(R)
    x[3:6] = R @ vel_b
    for _ in range(int(t / 0.005)):
        x, _, _ = af.step(x, np.array([0, 0, 0, duty, 0]), np.zeros(3), 0.005)
        x[0:3] = [0, 0, -60]
        x[3:6] = R @ vel_b
        x[6:10] = R2q(R)
        x[10:13] = 0
    return x


def accel(x, cmd, duty):
    xd, f = af.deriv(x, np.array([*cmd, duty, 0]), np.zeros(3))
    return xd[10:13], xd[3:6]


for name, R, vb, duty in (("HOVER (nose up)", euler2R(0, np.pi / 2, 0), np.zeros(3), 0.55),
                          ("CRUISE 13 m/s", euler2R(0, 0.06, 0), np.array([13.0, 0, 0.8]), 0.5)):
    x = settle_rotor(duty, vb, R)
    base, a0 = accel(x, [0, 0, 0], duty)
    print(name, "baseline ang accel", np.round(base, 2), " lin accel", np.round(a0, 2), "T=%.1f" % af.info['T'])
    for i, nm in enumerate(("aileron", "elevator", "rudder")):
        c = [0, 0, 0]
        c[i] = 0.5
        a1, _ = accel(x, c, duty)
        c[i] = -0.5
        a2, _ = accel(x, c, duty)
        print(f"   {nm:9s} authority = {(a1[i] - a2[i]) / 1.0:7.1f} rad/s^2 per unit cmd   "
              f"(cross: {np.round((a1 - a2) / 1.0, 1)})")
