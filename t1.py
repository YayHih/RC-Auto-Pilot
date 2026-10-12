import sys, time
import numpy as np
from rcsim.world import Sim
from rcsim.mathutil import R2euler, q2R

seq = sys.argv[1].split(',') if len(sys.argv) > 1 else ['takeoff']
total = float(sys.argv[2]) if len(sys.argv) > 2 else 24
s = Sim(seed=3)
print("settled z", round(s.x[2], 3), "est eul(deg)", np.round(np.degrees(s.est.euler()), 1),
      "true", np.round(np.degrees(R2euler(q2R(s.x[6:10]))), 1))
t0 = time.time()
for c in seq:
    s.press(c)
for k in range(int(total / 2)):
    s.run(2.0)
    L = s.log[-1]
    print(f"t={L['t']:5.1f} {L['state']:9s} pos={np.round(L['pos'], 1)} V={np.linalg.norm(L['vel']):5.1f} "
          f"eul={np.round(np.degrees(L['eul']), 0)} eeul={np.round(np.degrees(L['eeul']), 0)} "
          f"thr={L['thr']:.2f} crash={L['crashed']}")
    if L['crashed']:
        break
print("wall", round(time.time() - t0, 1))
