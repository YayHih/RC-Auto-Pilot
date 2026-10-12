"""Hover controller in isolation: start in a still, vertical hover at 40 m and let it run."""
import sys
import numpy as np
from rcsim.world import Sim
from rcsim.mathutil import euler2R, R2q, q2R, R2euler, qmul, qconj

s = Sim(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
R0 = euler2R(0.0, np.radians(88), 0.0)
s.x[0:3] = [0, 0, -40.0]
s.x[3:6] = 0
s.x[6:10] = R2q(R0)
s.x[13] = 700.0
s.duty = 0.55
s.est.q = R2q(R0).copy()
s.est.x[:, 0] = [0, 0, -40.0]
s.est.x[:, 1] = 0
s.est.alt_baro0 = s.est.alt_baro0 - 40.0 + 0.0
s.ap.state = 'HOVER'
s.ap.hover_tgt = np.array([0.0, 0.0, 0.0])
s.ap.alt_tgt = 40.0
s.ap.hdg_hold = 0.0
s.ap.th_cmd = np.pi / 2
s.ap.t_state = 0.0
T = float(sys.argv[1])
step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
n = 0
while s.t < T and not s.af.crashed:
    s.run(0.05)
    n += 1
    if n % int(step / 0.05) == 0:
        R = q2R(s.x[6:10])
        tilt = np.degrees(np.arccos(np.clip(-R[2, 0], -1, 1)))
        lean_dir = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        Rd = R_from = None
        qe = qmul(qconj(s.est.q), R2q(s.ap_des)) if hasattr(s, 'ap_des') else np.zeros(4)
        print(f"t={s.t:5.2f} pos={np.round(s.x[0:3],1)} v={np.round(s.x[3:6],1)} tilt={tilt:5.1f}@{lean_dir:4.0f} "
              f"w={np.round(s.x[10:13],2)} surf={np.round(s.servo,2)} thr={s.duty:.2f} T={s.T:.1f} "
              f"est_tilt={np.degrees(np.arccos(np.clip(-s.est.R[2,0],-1,1))):4.1f}")
print(s.af.crash_reason)
