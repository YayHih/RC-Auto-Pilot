"""Attitude-error trace during TO_HOVER / HOVER: nose angle from vertical, belly azimuth, errors."""
import sys
import numpy as np
from rcsim.world import Sim
from rcsim.mathutil import euler2R, q2R, R2q, qmul, qconj

s = Sim(seed=3)
for c in sys.argv[2].split(','):
    s.press(c)
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0
while s.t < float(sys.argv[1]) and not s.af.crashed:
    s.run(0.05)
    ap = s.ap
    if ap.state in ('TO_HOVER', 'HOVER') and s.n % 10 == 0 and s.t > t0:
        R = q2R(s.x[6:10])
        Rd = euler2R(ap.hdg_hold, ap.th_cmd, 0.0)
        nose = R[:, 0]
        nose_d = Rd[:, 0]
        tilt = np.degrees(np.arccos(np.clip(-nose[2], -1, 1)))             # from vertical
        ang_n = np.degrees(np.arccos(np.clip(nose @ nose_d, -1, 1)))
        belly_az = np.degrees(np.arctan2(R[1, 2], R[0, 2]))
        hdir = np.array([np.cos(ap.hdg_hold), np.sin(ap.hdg_hold), 0.0])
        lean = nose @ hdir          # + = nose leaning toward the heading, - = past vertical
        qe = qmul(qconj(s.x[6:10]), R2q(Rd))
        if qe[0] < 0:
            qe = -qe
        e = 2 * qe[1:]
        vb = R.T @ (s.x[3:6] - s.wind_now)
        alpha = np.degrees(np.arctan2(vb[2], vb[0]))
        beta = np.degrees(np.arcsin(np.clip(vb[1] / max(np.linalg.norm(vb), 0.1), -1, 1)))
        print(f"t={s.t:5.2f} {ap.state:8s} alpha={alpha:5.0f} beta={beta:5.0f} tilt={tilt:5.1f} lean={lean:+.2f} nose_err={ang_n:5.1f} belly_az={belly_az:6.0f} "
              f"(des {np.degrees(ap.hdg_hold):4.0f}) err_b={np.round(np.degrees(e),0)} "
              f"w={np.round(s.x[10:13],2)} surf={np.round(s.servo,2)} V={np.linalg.norm(s.x[3:6]):.1f} T={s.T:.1f} "
              f"alt={-s.x[2]:.0f}")
