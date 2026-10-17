"""Fly many different aeroplanes through a mission in parallel and score them.

    .venv/bin/python tools/batch.py [n_planes] [--seed S] [--mission full|short] [--cfg k=v,...] [--workers N]

Every plane is randomly different (size, control-surface size, per-axis control power, thrust, mass, wing
area) and flies in random wind/turbulence; the flight controller is not told.  Uses one worker per
performance core by default (Apple silicon: hw.perflevel0.physicalcpu) so the runs stay on the P cores.
"""
import os
import subprocess
import sys
import time
import multiprocessing as mp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

MISSIONS = {
    'full': ([(0, 'takeoff'), (25, 'loop'), (45, 'hover'), (70, 'loop'), (95, 'hover'), (115, 'land')], 300.0),
    'short': ([(0, 'takeoff'), (22, 'loop'), (38, 'hover'), (55, 'land')], 260.0),
    'land': ([(0, 'takeoff'), (20, 'land')], 150.0),
}
MORPH = 0.0          # mean interval of random in-flight changes (s), 0 = off; --morph S
RWY_X = (-75.0, 135.0)
RWY_HALF_W = 5.5
ON_RUNWAY_STATES = ('INIT', 'GROUND', 'TAKEOFF', 'FINAL', 'FLARE', 'ROLLOUT', 'GO_AROUND')


def perf_cores():
    try:
        return int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel0.physicalcpu']).strip())
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def random_plane(rng):
    return dict(scale=rng.uniform(0.85, 1.2), surf=rng.uniform(0.7, 1.4), ail=rng.uniform(0.8, 1.25),
                ele=rng.uniform(0.8, 1.25), rud=rng.uniform(0.8, 1.25), thrust=rng.uniform(0.8, 1.25),
                mass=rng.uniform(0.9, 1.2), wing=rng.uniform(0.85, 1.15))


def fly(job):
    """One mission. job = (plane dict, sim seed, mission name, cfg overrides dict[, morph interval s])."""
    plane, seed, mission, overrides = job[:4]
    morph = job[4] if len(job) > 4 else MORPH
    from rcsim.world import Sim
    from rcsim import fcbridge as fb
    cfg = fb.default_config()
    for k, v in (overrides or {}).items():
        cur = getattr(cfg, k)
        if hasattr(cur, '__len__'):
            for i, x in enumerate(v):
                cur[i] = x
        else:
            setattr(cfg, k, type(cur)(v))
    sched, tmax = MISSIONS[mission]
    s = Sim(seed=seed, plane=plane, cfg=cfg, morph=morph)
    side = cfg.fly_side
    i = 0
    worst_pilot_side = 0.0          # how far onto the pilot's side of the runway edge, while airborne
    worst_box = 1e9                 # closest to the centreline while manoeuvring (orbit/loop/hover/circle)
    ps_where = box_where = ''
    gas = 0
    last = None
    td = None
    while s.t < tmax and not s.af.crashed:
        while i < len(sched) and s.t >= sched[i][0]:
            s.press(sched[i][1])
            i += 1
        s.run(0.1)
        st = s.ap.state
        y = side * s.x[1]           # runway points north from the origin: east = +y
        alt = -s.x[2]
        if alt > 2.0:
            if -RWY_HALF_W - y > worst_pilot_side:
                worst_pilot_side, ps_where = -RWY_HALF_W - y, f"{st}@({s.x[0]:.0f},{s.x[1]:.0f},{alt:.0f}m)"
            if st not in ON_RUNWAY_STATES and y < worst_box:
                worst_box, box_where = y, st
        if st == 'GO_AROUND' and last != 'GO_AROUND':
            gas += 1
        if st == 'ROLLOUT' and td is None:
            td = s.x[0:2].copy()
        last = st
        if st == 'GROUND' and s.t > sched[-1][0] + 5:
            break
    x, y = s.x[0], s.x[1]
    on_rwy = RWY_X[0] < x < RWY_X[1] and abs(y) < RWY_HALF_W
    ok = (not s.af.crashed) and s.ap.state == 'GROUND' and on_rwy and s.t > sched[-1][0]
    return dict(plane=plane, seed=seed, ok=ok, crashed=s.af.crashed, reason=s.af.crash_reason, state=s.ap.state,
                t=s.t, pos=(float(x), float(y)), td=None if td is None else (float(td[0]), float(td[1])),
                go_arounds=gas, pilot_side=float(worst_pilot_side), box=float(worst_box), ps_where=ps_where, box_where=box_where,
                auth=[round(a, 2) for a in s.fout.auth_q[:]],
                morphs=len(s.morpher.events) if s.morpher else 0, morph_now=dict(s.af.morph))


def _fly_idx(arg):
    i, job = arg
    return i, fly(job)


def run_batch(jobs, workers=None, progress=None):
    """Results in job order; progress(i, result) is called as each flight finishes."""
    workers = workers or perf_cores()
    ctx = mp.get_context('fork')
    out = [None] * len(jobs)
    with ctx.Pool(workers) as pool:
        for i, r in pool.imap_unordered(_fly_idx, list(enumerate(jobs)), chunksize=1):
            out[i] = r
            if progress:
                progress(i, r)
    return out


def line(r):
    p = r['plane']
    return (f"{'OK ' if r['ok'] else 'BAD'} size x{p['scale']:.2f} surf x{p['surf']:.2f} thrust x{p['thrust']:.2f} "
            f"mass x{p['mass']:.2f} wing x{p['wing']:.2f}: {r['state']:9s} pos ({r['pos'][0]:6.1f},{r['pos'][1]:5.1f}) "
            f"GA {r['go_arounds']} pilot-side {r['pilot_side']:4.1f} m {r['ps_where']}  box {r['box']:5.1f} m {r['box_where']}  "
            f"auth {r['auth']}" + (f"  morphs {r['morphs']} now " + '/'.join(f"{v:.2f}" for v in r['morph_now'].values())
                                    if r.get('morphs') else '') + (f"  [{r['reason']}]" if r['crashed'] else ''))


def make_jobs(n, seed, mission, overrides=None):
    rng = np.random.default_rng(seed)
    return [(random_plane(rng), int(rng.integers(1, 10**6)), mission, overrides) for _ in range(n)]


def summary(res):
    n = len(res)
    ok = sum(r['ok'] for r in res)
    crashes = sum(r['crashed'] for r in res)
    lat = [abs(r['pos'][1]) for r in res if r['ok']]
    viol = sum(r['pilot_side'] > 0.0 for r in res)
    box = [r['box'] for r in res if r['box'] < 1e8]
    return (f"{ok}/{n} landed and stopped on the runway, {crashes} crashed, go-arounds {sum(r['go_arounds'] for r in res)}, "
            f"lateral error mean {np.mean(lat) if lat else float('nan'):.1f} m, pilot-side incursions {viol}, "
            f"closest manoeuvre to centreline {min(box) if box else float('nan'):.0f} m")


if __name__ == '__main__':
    def opt(name, default, cast=str):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = cast(sys.argv[i + 1])
            del sys.argv[i:i + 2]
            return v
        return default
    seed = opt('--seed', 1, int)
    mission = opt('--mission', 'full')
    workers = opt('--workers', None, int)
    MORPH = opt('--morph', 0.0, float)
    ov = opt('--cfg', '')
    overrides = {k: float(v) for k, v in (kv.split('=') for kv in ov.split(','))} if ov else None
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    t0 = time.time()
    res = run_batch(make_jobs(n, seed, mission, overrides), workers,
                    progress=lambda i, r: print(f"#{i:2d} " + line(r), flush=True))
    print(summary(res))
    print(f"[{len(res)} flights on {workers or perf_cores()} performance-core workers, {time.time() - t0:.0f} s]")
