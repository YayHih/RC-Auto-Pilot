"""Headless scripted flight:  python scenario.py "0:takeoff,20:loop,45:hover,70:land" 140 [seed] [plot.png]
                                  [--wind M/S] [--wind-from DEG] [--turb X] [--gusts X] [--no-random]
                                  [--fc native|arm|lockstep] [--plane ail=0.6,ele=1.4,rud=1,mass=1.2,thrust=1]
                                  [--cfg adapt_enable=0,rwy_hdg_deg=0,...]   (fc_config_t fields)
Wind, gusts and airframe tolerances are drawn from the seed unless given; --no-random = nominal airframe."""
import sys
import numpy as np
from rcsim.world import Sim


def _pop(name, cast=float, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


_kw = dict(wind=_pop('--wind'), wind_from=_pop('--wind-from'), turb=_pop('--turb', default=1.0),
           gusts=_pop('--gusts', default=1.0), fc=_pop('--fc', str, 'native'))
_cfg = _pop('--cfg', str)
if _cfg:        # flight-controller config overrides, e.g. --cfg adapt_enable=0,rwy_hdg_deg=0
    from rcsim import fcbridge as _fb
    _c = _fb.default_config()
    for kv in _cfg.split(','):
        k, v = kv.split('=')
        setattr(_c, k, type(getattr(_c, k))(float(v)) if not isinstance(getattr(_c, k), int) else int(float(v)))
    _kw['cfg'] = _c
_plane = _pop('--plane', str)
if _plane:      # e.g. --plane ail=0.6,ele=1.4,mass=1.2
    _kw['plane'] = {k: float(v) for k, v in (kv.split('=') for kv in _plane.split(','))}
if '--no-random' in sys.argv:
    sys.argv.remove('--no-random')
    _kw['randomize'] = 0.0

sched = [(float(a), b) for a, b in (x.split(':') for x in sys.argv[1].split(','))]
tmax = float(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
png = sys.argv[4] if len(sys.argv) > 4 else None
s = Sim(seed=seed, **_kw)
print(f"wind {s.atm.W20:.1f} m/s from {s.atm.from_deg:.0f} deg, turb {s.atm.turb:.1f}, gusts {s.atm.gusts:.1f}")
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
air = [l for l in s.log if l['state'] in ('CRUISE', 'LAND', 'LOOP_PREP')]
if air:
    we = np.array([l['wind_est'] for l in air[-200:]]).mean(0)
    wt = np.array([l['wind_true'][:2] for l in air[-200:]]).mean(0)
    print(f"wind est {np.hypot(*we):.1f} m/s from {np.degrees(np.arctan2(-we[1], -we[0])) % 360:.0f}  "
          f"true {np.hypot(*wt):.1f} m/s from {np.degrees(np.arctan2(-wt[1], -wt[0])) % 360:.0f}")
print("learned authority  airspeed part", np.round(L['auth_q'], 2), " prop-wash part", np.round(L['auth_w'], 2))
if s.cycle_log:
    c = np.array([x[2] for x in s.cycle_log])
    print(f"M0+ cycles/tick mean {c.mean():.0f} max {c.max():.0f} -> {c.max() / 48e3:.2f} ms of the 10 ms tick")

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
