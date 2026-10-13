"""Flight-controller checks:  .venv/bin/python tests/fc_check.py [seconds] [seed]

1. builds the Cortex-M0+ firmware core with the CubeCell toolchain (same flags as the Arduino build)
   and reports flash / RAM / stack use against the ASR6502's 128 KB / 16 KB
2. flies a full mission with the ARM machine code (Unicorn) and the PC build in lockstep: every one of
   the 100 Hz ticks must give bit-identical outputs (PCA9685 counts and all telemetry)
3. reports the cycle count per tick in every flight state at 48 MHz, plus the I2C time, against the
   10 ms tick
"""
import os
import sys
import time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rcsim import fcbridge as fb          # noqa: E402
from rcsim.world import Sim               # noqa: E402

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ok = True

fb.build_arm()
sz = fb.arm_section_sizes()
text = sz.get('.text', (0, 0))[0]
ram = sz.get('.data', (0, 0))[0] + sz.get('.bss', (0, 0))[0]
su = fb.arm_stack_usage()
print(f"flight-controller core on the M0+: flash {text / 1024:.1f} KB, static RAM {ram} B, "
      f"largest stack frame {max(su.values())} B ({max(su, key=su.get)})")

s = Sim(seed=seed, fc='lockstep', cycle_every=5)
sched = [(0, 'takeoff'), (25, 'loop'), (45, 'hover'), (70, 'loop'), (95, 'hover'), (115, 'land')]
i = 0
t0 = time.time()
try:
    while s.t < secs and not s.af.crashed:
        while i < len(sched) and s.t >= sched[i][0]:
            s.press(sched[i][1])
            i += 1
        s.run(0.5)
except AssertionError as e:
    ok = False
    print('FAIL', e)
n = s.fc.n
print(f"lockstep: {n} ticks ({n / 100:.0f} s of flight, ended in {s.ap.state}) bit-identical ARM vs PC: "
      f"{'yes' if ok else 'NO'}   [{time.time() - t0:.0f} s wall]")
print(f"stack high-water in the emulator: {s.fc.arm.stack_high_water()} B")

by = defaultdict(list)
for _, st, c in s.cycle_log:
    by[st].append(c)
io_us = Sim.IO_READ_US + Sim.IO_WRITE_US
print(f"\ncycles per 10 ms tick at 48 MHz (+{io_us} us of I2C):")
worst = 0
for st, c in sorted(by.items(), key=lambda kv: -max(kv[1])):
    c = np.array(c)
    ms = c.max() / 48e3 + io_us / 1e3
    worst = max(worst, ms)
    print(f"  {st:10s} mean {c.mean():7.0f}  max {c.max():7.0f}  -> worst tick {ms:4.2f} ms ({ms * 10:.0f} % of the tick)")
ok &= worst < 10.0
print('\nALL PASS' if ok else '\nFAILED')
