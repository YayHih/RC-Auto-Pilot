"""Open-loop: nose pointing near vertical, climbing, no control. Is it stable in pitch and yaw?"""
import numpy as np
from rcsim.airframe import Airframe
from rcsim.mathutil import euler2R, R2q, q2R

af = Airframe()
for label, R0 in (("tilt 10deg BACK (pitch past vertical)", euler2R(0, np.radians(100), 0)),
                  ("tilt 10deg FORWARD", euler2R(0, np.radians(80), 0)),
                  ("tilt 10deg SIDEWAYS (about body z)", euler2R(0, np.radians(90), 0) @ np.array(
                      [[np.cos(.17), -np.sin(.17), 0], [np.sin(.17), np.cos(.17), 0], [0, 0, 1]]))):
    x = np.zeros(15)
    x[2] = -80
    x[6:10] = R2q(R0)
    x[3:6] = np.array([0, 0, -15.0])        # climbing straight up at 15 m/s
    x[13] = 700.0                             # rotor
    out = []
    for i in range(int(1.2 / 0.005)):
        x, _, _ = af.step(x, np.array([0, 0, 0, 0.55, 0]), np.zeros(3), 0.005)
        if i % 40 == 0:
            R = q2R(x[6:10])
            tilt = np.degrees(np.arccos(np.clip(-R[2, 0], -1, 1)))
            out.append(f"{i*0.005:.1f}s tilt={tilt:5.1f} w={np.round(x[10:13],2)} vz={-x[5]:.1f}")
    print(label)
    print("   " + "\n   ".join(out))
