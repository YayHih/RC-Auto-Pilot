"""Headless scripted flight:  python scenario.py "0:takeoff,20:loop,45:hover,70:land" 140 [seed] [plot.png]"""
import sys
import numpy as np
from rcsim.world import Sim

sched = [(float(a), b) for a, b in (x.split(':') for x in sys.argv[1].split(','))]
tmax = float(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
png = sys.argv[4] if len(sys.argv) > 4 else None
s = Sim(seed=seed)
last_state, i = None, 0
while s.t < tmax and not s.af.crashed:
    while i < len(sched) and s.t >= sched[i][0]:
        print(f"[{s.t:6.1f}s] BUTTON {sched[i][1]}")
        s.press(sched[i][1])
        i += 1
    s.run(0.05)
    L = s.log[-1]
    if L['state'] != last_state:
        print(f"[{L['t']:6.1f}s] -> {L['state']:10s} pos={np.round(L['pos'], 0)} V={np.linalg.norm(L['vel']):.1f}")
        last_state = L['state']
print("END t=%.1f state=%s crashed=%s %s" % (s.t, s.ap.state, s.af.crashed, s.af.crash_reason))
L = s.log[-1]
print("final pos", np.round(L['pos'], 1), "speed", round(float(np.linalg.norm(L['vel'])), 2))

if png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array([l['t'] for l in s.log])
    P = np.array([l['pos'] for l in s.log])
    E = np.array([l['epos'] for l in s.log])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(P[:, 1], P[:, 0], label='true'); ax[0, 0].plot(E[:, 1], E[:, 0], '--', label='estimated')
    ax[0, 0].set_xlabel('East m'); ax[0, 0].set_ylabel('North m'); ax[0, 0].axis('equal'); ax[0, 0].legend()
    ax[0, 1].plot(t, -P[:, 2], label='true'); ax[0, 1].plot(t, -E[:, 2], '--', label='estimated')
    ax[0, 1].set_ylabel('Altitude m'); ax[0, 1].set_xlabel('s'); ax[0, 1].legend()
    ax[1, 0].plot(t, np.linalg.norm(P - E, axis=1)); ax[1, 0].set_ylabel('position est error m')
    ax[1, 1].plot(t, [np.linalg.norm(l['vel']) for l in s.log]); ax[1, 1].set_ylabel('speed m/s')
    plt.tight_layout(); plt.savefig(png, dpi=90)
    print("saved", png)
