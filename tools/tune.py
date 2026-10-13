"""Hyper-parameter search for the flight controller over many different aeroplanes.

    .venv/bin/python tools/tune.py [n_planes] [n_random] [n_local] [--mission short] [--seed S]

Every candidate flies the same set of random aeroplanes (common random numbers, so candidates are
compared on equal terms) on the performance cores.  Cost (lower is better):
    10 x failed landings + 5 x crashes + 3 x pilot-side incursions + 0.5 x go-arounds
    + 0.1 x lateral stop error (m) + 0.002 x final-approach length (m; prefers a compact circuit)
Random search over the box, then a local search around the best.  Writes tools/tune_result.json.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from batch import make_jobs, run_batch, perf_cores   # noqa: E402

DEFAULT = dict(v_approach=12.3, glide_deg=8.4, pattern_alt_m=12.5, hold_r_m=37.5, flare_h_m=1.5, flare_tau_s=1.2,
               bw_fwd_k=1.0, bw_hover_k=1.0, att_kp=5.2)          # = fc_default_config (result of the last search)
BOX = dict(v_approach=(9.5, 12.5), glide_deg=(5.0, 9.0), pattern_alt_m=(12.0, 20.0), hold_r_m=(28.0, 45.0),
           flare_h_m=(1.0, 2.5), flare_tau_s=(1.0, 2.5), bw_fwd_k=(0.7, 1.4), bw_hover_k=(0.7, 1.3), att_kp=(5.0, 11.0))
BW_FWD, BW_HOVER = (20.8, 18.2, 18.2), (15.4, 30.7, 30.7)


def to_cfg(p):
    d = {k: v for k, v in p.items() if not k.endswith('_k')}
    d['bw_fwd'] = [b * p['bw_fwd_k'] for b in BW_FWD]
    d['bw_hover'] = [b * p['bw_hover_k'] for b in BW_HOVER]
    return d


def cost(res, p):
    n = len(res)
    fail = sum(not r['ok'] for r in res) / n
    crash = sum(r['crashed'] for r in res) / n
    pilot = sum(r['pilot_side'] > 0 for r in res) / n
    ga = sum(r['go_arounds'] for r in res) / n
    lat = np.mean([min(abs(r['pos'][1]), 20.0) for r in res])
    final_len = p['pattern_alt_m'] / np.tan(np.radians(p['glide_deg']))
    return 10 * fail + 5 * crash + 3 * pilot + 0.5 * ga + 0.1 * lat + 0.002 * final_len


def evaluate(p, jobs, log):
    ov = to_cfg(p)
    js = [(pl, sd, m, ov) for pl, sd, m, _ in jobs]
    t0 = time.time()
    res = run_batch(js)
    c = cost(res, p)
    ok = sum(r['ok'] for r in res)
    rec = dict(params=p, cost=c, ok=ok, n=len(res), crashed=sum(r['crashed'] for r in res),
               go_arounds=sum(r['go_arounds'] for r in res), secs=time.time() - t0)
    log.append(rec)
    print(f"cost {c:6.3f}  ok {ok}/{len(res)}  crashes {rec['crashed']}  GA {rec['go_arounds']}  "
          + ' '.join(f"{k}={v:.2f}" for k, v in p.items()) + f"  [{rec['secs']:.0f} s]", flush=True)
    return c


if __name__ == '__main__':
    def opt(name, default, cast=str):
        if name in sys.argv:
            i = sys.argv.index(name)
            v = cast(sys.argv[i + 1])
            del sys.argv[i:i + 2]
            return v
        return default
    mission = opt('--mission', 'short')
    seed = opt('--seed', 7, int)
    n_planes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    n_local = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rng = np.random.default_rng(seed)
    jobs = make_jobs(n_planes, seed, mission)
    print(f"{n_planes} planes x {mission} mission per candidate on {perf_cores()} performance cores", flush=True)
    log = []
    best = dict(DEFAULT)
    best_c = evaluate(best, jobs, log)
    for _ in range(n_random):
        p = {k: float(rng.uniform(*BOX[k])) for k in BOX}
        c = evaluate(p, jobs, log)
        if c < best_c:
            best, best_c = p, c
    step = 0.25
    for _ in range(n_local):
        p = {k: float(np.clip(v + step * (BOX[k][1] - BOX[k][0]) * rng.normal() * 0.5, *BOX[k])) for k, v in best.items()}
        c = evaluate(p, jobs, log)
        if c < best_c:
            best, best_c = p, c
        else:
            step = max(step * 0.85, 0.05)
    out = dict(best=best, best_cost=best_c, default_cost=log[0]['cost'], cfg=to_cfg(best), log=log)
    with open(os.path.join(os.path.dirname(__file__), 'tune_result.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('BEST', best_c, best)
