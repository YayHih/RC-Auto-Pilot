import sys
import numpy as np
from rcsim.world import Sim
from rcsim.mathutil import R2euler, q2R

seq = sys.argv[1].split(',')
state = sys.argv[2]                 # print rows only while in this state
tmax = float(sys.argv[3])
every = int(float(sys.argv[4]) / 0.05) if len(sys.argv) > 4 else 5
s = Sim(seed=3)
sched = sorted((float(c.split('@')[1]) if '@' in c else 0.0, c.split('@')[0]) for c in seq)
n = 0
while s.t < tmax and not s.af.crashed:
    while sched and s.t >= sched[0][0]:
        s.press(sched.pop(0)[1])
    s.run(0.05)
    L = s.log[-1]
    n += 1
    if L['state'] == state and n % every == 0:
        ap = s.ap
        print(f"t={L['t']:5.1f} thcmd={np.degrees(ap.th_cmd):6.0f} true_eul={np.round(np.degrees(L['eul']),0)} "
              f"pos={np.round(L['pos'],0)} V={np.linalg.norm(L['vel']):4.1f} w={np.round(s.x[10:13],2)} "
              f"surf={np.round(L['surf'],2)} thr={L['thr']:.2f}")
print("end", s.ap.state, "crashed", s.af.crashed, "t", round(s.t, 1))
