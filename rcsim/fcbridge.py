"""Bridge to the C++ flight controller in firmware/RCAutopilot/src/fc.

Two back-ends run the *same source*:
  NativeFC  compiled for this PC (clang, FMA contraction off) and called through ctypes - fast.
  ArmFC     compiled with the CubeCell toolchain for the Cortex-M0+ (same flags as the Arduino build)
            and executed instruction by instruction in Unicorn.  Gives the real code's cycle count
            per tick at 48 MHz and proves the PC build produces bit-identical servo outputs.
The math library in the firmware avoids libm, so both produce the same bits (tests/fc_check.py).
"""
import ctypes as C
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'firmware', 'RCAutopilot', 'src', 'fc')
BUILD = os.path.join(ROOT, 'build')
TOOLS = os.path.join(ROOT, 'tools')

STATE_NAMES = ['INIT', 'GROUND', 'TAKEOFF', 'CRUISE', 'LOOP_PREP', 'LOOP', 'TO_HOVER', 'HOVER',
               'FROM_HOVER', 'LAND', 'FINAL', 'FLARE', 'ROLLOUT', 'GO_AROUND', 'MANUAL']
CMDS = {'takeoff': 1, 'hover': 2, 'loop': 3, 'land': 4, 'cruise': 5}
CMD_NAMES = {v: k for k, v in CMDS.items()}
RC_AUTO, RC_DIRECT, RC_ASSIST = 0, 1, 2
IN_IMU, IN_MAG, IN_BARO, IN_GPS = 1, 2, 4, 8


class FcIn(C.Structure):
    _fields_ = [('t_us', C.c_uint32), ('flags', C.c_uint8), ('cmd', C.c_uint8), ('rc_mode', C.c_uint8),
                ('_pad0', C.c_uint8), ('acc', C.c_int16 * 3), ('gyro', C.c_int16 * 3), ('mag', C.c_int16 * 3),
                ('rc_us', C.c_int16 * 4), ('baro_q8', C.c_uint32), ('lat_e7', C.c_int32), ('lon_e7', C.c_int32),
                ('hmsl_mm', C.c_int32), ('vel_mm_s', C.c_int32 * 3), ('hacc_mm', C.c_uint32),
                ('fix_type', C.c_uint8), ('num_sv', C.c_uint8), ('_pad1', C.c_uint16)]


class FcOut(C.Structure):
    _fields_ = [('pca_off', C.c_uint16 * 8), ('state', C.c_uint8), ('goal', C.c_uint8), ('rc_mode', C.c_uint8),
                ('flags', C.c_uint8), ('pos', C.c_float * 3), ('vel', C.c_float * 3), ('q', C.c_float * 4),
                ('rates', C.c_float * 3), ('wind', C.c_float * 2), ('wind_sd', C.c_float), ('airspeed', C.c_float),
                ('auth_q', C.c_float * 3), ('auth_w', C.c_float * 3), ('cmd', C.c_float * 5),
                ('land_dir', C.c_float), ('rwy_hdg', C.c_float), ('tick', C.c_uint32)]


class FcConfig(C.Structure):
    _fields_ = [('pca_osc_hz', C.c_float), ('pca_prescale', C.c_uint8), ('_pad0', C.c_uint8 * 3),
                ('ch_center_us', C.c_int16 * 8), ('ch_throw_us', C.c_int16 * 8), ('ch_reverse', C.c_int8 * 8),
                ('mag_decl_deg', C.c_float), ('mag_hard_iron_uT', C.c_float * 3), ('mag_motor_uT', C.c_float * 3),
                ('auth_q0', C.c_float * 3), ('auth_w0', C.c_float * 3), ('auth_lockin_cal', C.c_float * 3), ('auth_lockin_cal_w', C.c_float * 3),
                ('thrust_max_n', C.c_float),
                ('weight_n', C.c_float), ('thr_hover', C.c_float), ('rwy_len_ahead_m', C.c_float),
                ('rwy_len_behind_m', C.c_float), ('rwy_hdg_deg', C.c_float), ('cruise_alt_m', C.c_float), ('cruise_v', C.c_float),
                ('pattern_alt_m', C.c_float), ('v_approach', C.c_float), ('glide_deg', C.c_float),
                ('fence_m', C.c_float), ('hold_r_m', C.c_float), ('flare_h_m', C.c_float),
                ('flare_tau_s', C.c_float), ('bw_fwd', C.c_float * 3), ('bw_hover', C.c_float * 3),
                ('att_kp', C.c_float),
                ('adapt_enable', C.c_uint8), ('brake_enable', C.c_uint8), ('fly_side', C.c_int8), ('_pad1', C.c_uint8)]


def _sizes_word():
    return C.sizeof(FcIn) | (C.sizeof(FcOut) << 10) | (C.sizeof(FcConfig) << 20)


def _sources():
    return [os.path.join(SRC, f) for f in sorted(os.listdir(SRC)) if f.endswith(('.cpp', '.h'))]


def _stale(target, deps):
    if not os.path.exists(target):
        return True
    t = os.path.getmtime(target)
    return any(os.path.getmtime(d) > t for d in deps)


# ------------------------------------------------------------------------------------- native
def build_native():
    os.makedirs(BUILD, exist_ok=True)
    ext = '.dylib' if sys.platform == 'darwin' else ('.dll' if os.name == 'nt' else '.so')
    lib = os.path.join(BUILD, 'libfc' + ext)
    srcs = _sources()
    if _stale(lib, srcs):
        cxx = shutil.which('clang++') or shutil.which('g++') or 'c++'
        tmp = lib + '.%d.tmp' % os.getpid()
        cmd = [cxx, '-std=c++11', '-O2', '-Wall', '-Wdouble-promotion', '-Werror=double-promotion',
               '-ffp-contract=off', '-fno-fast-math', '-shared', '-fPIC', '-o', tmp] + \
              [s for s in srcs if s.endswith('.cpp')]
        subprocess.run(cmd, check=True)
        os.replace(tmp, lib)                 # atomic: parallel simulations never see half a library
    return lib


class NativeFC:
    """Flight controller compiled for this computer."""

    def __init__(self, cfg=None):
        self.lib = C.CDLL(build_native())
        self.lib.fc_tick.argtypes = [C.POINTER(FcIn), C.POINTER(FcOut)]
        self.lib.fc_init.argtypes = [C.POINTER(FcConfig)]
        self.lib.fc_default_config.argtypes = [C.POINTER(FcConfig)]
        self.lib.fc_struct_sizes.restype = C.c_uint32
        if self.lib.fc_struct_sizes() != _sizes_word():
            raise RuntimeError('fc_types.h and fcbridge.py struct layouts differ')
        self.cfg = cfg or self.default_config()
        self.out = FcOut()
        self.lib.fc_init(C.byref(self.cfg))
        self.cycles = None

    def default_config(self):
        c = FcConfig()
        self.lib.fc_default_config(C.byref(c))
        return c

    def tick(self, fin):
        self.lib.fc_tick(C.byref(fin), C.byref(self.out))
        return self.out


def default_config():
    lib = C.CDLL(build_native())
    c = FcConfig()
    lib.fc_default_config(C.byref(c))
    return c


# ------------------------------------------------------------------------------------- ARM
def _arm_bin():
    base = os.path.expanduser('~/Library/Arduino15/packages/CubeCell/tools/gcc-arm-none-eabi')
    for root in (base, os.path.expanduser('~/.arduino15/packages/CubeCell/tools/gcc-arm-none-eabi'),
                 os.path.expandvars(r'%LOCALAPPDATA%\Arduino15\packages\CubeCell\tools\gcc-arm-none-eabi')):
        if os.path.isdir(root):
            for v in sorted(os.listdir(root)):
                b = os.path.join(root, v, 'bin')
                if os.path.exists(os.path.join(b, 'arm-none-eabi-g++')) or os.path.exists(os.path.join(b, 'arm-none-eabi-g++.exe')):
                    return b
    p = shutil.which('arm-none-eabi-g++')
    if p:
        return os.path.dirname(p)
    raise RuntimeError('CubeCell arm-none-eabi toolchain not found (install the CubeCell board package)')


ARM_CFLAGS = ['-mcpu=cortex-m0plus', '-mthumb', '-Os', '-ffunction-sections', '-fdata-sections', '-fno-common',
              '-fno-exceptions', '-fno-rtti', '-ffp-contract=off', '-Wall', '-Wdouble-promotion',
              '-Werror=double-promotion', '-fstack-usage']


def build_arm():
    """Cortex-M0+ build of the flight controller + a 3-function harness, linked for the emulator."""
    os.makedirs(BUILD, exist_ok=True)
    elf = os.path.join(BUILD, 'fc_arm.elf')
    harness = os.path.join(TOOLS, 'arm_harness.cpp')
    ld = os.path.join(TOOLS, 'emu.ld')
    deps = _sources() + [harness, ld]
    if _stale(elf, deps):
        b = _arm_bin()
        gxx = os.path.join(b, 'arm-none-eabi-g++')
        objs = []
        for s in [x for x in _sources() if x.endswith('.cpp')] + [harness]:
            o = os.path.join(BUILD, 'arm_' + os.path.basename(s) + '.o')
            subprocess.run([gxx] + ARM_CFLAGS + ['-I', SRC, '-c', s, '-o', o], check=True, cwd=BUILD)
            objs.append(o)
        subprocess.run([gxx, '-mcpu=cortex-m0plus', '-mthumb', '-nostartfiles', '-specs=nano.specs',
                        '-Wl,--gc-sections', '-Wl,-T,' + ld, '-Wl,-Map,' + os.path.join(BUILD, 'fc_arm.map'),
                        '-o', elf] + objs + ['-lgcc', '-lc'], check=True)
        objcopy = os.path.join(b, 'arm-none-eabi-objcopy')
        subprocess.run([objcopy, '-O', 'binary', '-j', '.text', elf, os.path.join(BUILD, 'fc_arm_text.bin')], check=True)
        subprocess.run([objcopy, '-O', 'binary', '-j', '.data', elf, os.path.join(BUILD, 'fc_arm_data.bin')], check=True)
        nm = subprocess.run([os.path.join(b, 'arm-none-eabi-nm'), '-S', elf], check=True, capture_output=True, text=True).stdout
        with open(os.path.join(BUILD, 'fc_arm.sym'), 'w') as f:
            f.write(nm)
        size = subprocess.run([os.path.join(b, 'arm-none-eabi-size'), '-A', elf], check=True, capture_output=True, text=True).stdout
        with open(os.path.join(BUILD, 'fc_arm.size'), 'w') as f:
            f.write(size)
    return elf


def arm_symbols():
    syms = {}
    with open(os.path.join(BUILD, 'fc_arm.sym')) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                syms[p[3]] = (int(p[0], 16), int(p[1], 16))
            elif len(p) == 3:
                syms[p[2]] = (int(p[0], 16), 0)
    return syms


def arm_section_sizes():
    out = {}
    with open(os.path.join(BUILD, 'fc_arm.size')) as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0].startswith('.'):
                out[p[0]] = (int(p[1]), int(p[2]))
    return out


def arm_stack_usage():
    """Worst single-frame stack use per function from -fstack-usage (bytes)."""
    res = {}
    for fn in os.listdir(BUILD):
        if fn.startswith('arm_') and fn.endswith('.su'):
            with open(os.path.join(BUILD, fn)) as f:
                for line in f:
                    p = line.rsplit('\t', 2)
                    if len(p) == 3:
                        res[p[0].split(':')[-1]] = int(p[1])
    return res


class ArmFC:
    """The Cortex-M0+ machine code of the flight controller, run in Unicorn.

    cycle_every: count cycles on every Nth tick (block hook, slower); 0 = never.
    Cycle model (ARM Cortex-M0+ TRM): ALU 1, load/store 2, LDM/STM/PUSH/POP 1+N (POP pc 3+N),
    taken branch 2, BL 3, plus one flash wait state per taken branch (sequential fetches are covered
    by the flash read accelerator).  Peripheral I/O time (I2C) is added separately by the caller."""

    F_CPU = 48_000_000
    FLASH, RAM, RAM_SIZE = 0x00000000, 0x20000000, 0x10000
    RET = 0x0003FF00

    def __init__(self, cfg=None, cycle_every=0, flash_ws_per_branch=1):
        from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_BLOCK
        from unicorn import arm_const as A
        build_arm()
        self.A = A
        self.syms = arm_symbols()
        text = open(os.path.join(BUILD, 'fc_arm_text.bin'), 'rb').read()
        data = open(os.path.join(BUILD, 'fc_arm_data.bin'), 'rb').read()
        self.text = text
        mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        try:
            from unicorn import UC_CPU_ARM_CORTEX_M0
            mu.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M0)
        except Exception:
            pass
        mu.mem_map(self.FLASH, 0x40000)
        mu.mem_map(self.RAM, self.RAM_SIZE)
        mu.mem_write(self.FLASH, text)
        mu.mem_write(self.RET, b'\x00\xbf\x00\xbf')          # nop; nop
        if data:
            mu.mem_write(self.syms['__data_start'][0], data)
        # paint the stack so the high-water mark can be read back
        self.stack_top = self.RAM + self.RAM_SIZE
        self.stack_paint_from = self.stack_top - 0x2000
        mu.mem_write(self.stack_paint_from, b'\xa5' * 0x2000)
        self.mu = mu
        self.ws = flash_ws_per_branch
        self.cycle_every = cycle_every
        self._blk = {}
        self.cycles = None
        self.cycle_log = []
        self.n = 0
        self._counting = False
        self._acc = 0
        mu.hook_add(UC_HOOK_BLOCK, self._on_block)
        self.a_in = self.syms['fc_arm_in'][0]
        self.a_out = self.syms['fc_arm_out'][0]
        self.a_cfg = self.syms['fc_arm_cfg'][0]
        if self.syms['fc_arm_in'][1] != C.sizeof(FcIn) or self.syms['fc_arm_out'][1] != C.sizeof(FcOut):
            raise RuntimeError('ARM struct sizes differ from fcbridge.py')
        if cfg is None:
            self._call('fc_arm_defaults')
            cfg = FcConfig.from_buffer_copy(bytes(mu.mem_read(self.a_cfg, C.sizeof(FcConfig))))
        self.cfg = cfg
        mu.mem_write(self.a_cfg, bytes(cfg))
        self._call('fc_arm_init')
        self.out = FcOut()

    def default_config(self):
        return self.cfg

    # cycles for one basic block, decoded once from the Thumb code
    def _block_cycles(self, addr, size):
        cyc, i, end = 0, addr, addr + size
        t = self.text
        branch = False
        while i < end:
            hw = t[i] | (t[i + 1] << 8)
            if (hw >> 11) in (0x1D, 0x1E, 0x1F):              # 32-bit: BL, MRS/MSR, barriers
                h2 = t[i + 2] | (t[i + 3] << 8)
                cyc += 3 if (hw & 0xF800) == 0xF000 and (h2 & 0xD000) == 0xD000 else 4
                branch = True
                i += 4
                continue
            if (hw & 0xF800) == 0x4800 or (hw & 0xF000) in (0x5000, 0x6000, 0x7000, 0x8000, 0x9000):
                cyc += 2
            elif (hw & 0xF000) == 0xC000:
                cyc += 1 + bin(hw & 0xFF).count('1')
            elif (hw & 0xFE00) == 0xB400:
                cyc += 1 + bin(hw & 0x1FF).count('1')
            elif (hw & 0xFE00) == 0xBC00:
                n = bin(hw & 0xFF).count('1')
                cyc += (3 + n + 1) if hw & 0x100 else (1 + n)
                branch = branch or bool(hw & 0x100)
            elif (hw & 0xF000) == 0xD000 or (hw & 0xF800) == 0xE000 or (hw & 0xFF00) == 0x4700:
                cyc += 2
                branch = True
            else:
                cyc += 1
            i += 2
        return cyc + (self.ws if branch else 0)

    def _on_block(self, mu, addr, size, user):
        if not self._counting:
            return
        k = (addr, size)
        c = self._blk.get(k)
        if c is None:
            c = self._blk[k] = self._block_cycles(addr, size)
        self._acc += c

    def _call(self, name):
        A = self.A
        self.mu.reg_write(A.UC_ARM_REG_SP, self.stack_top)
        self.mu.reg_write(A.UC_ARM_REG_LR, self.RET | 1)
        self.mu.emu_start(self.syms[name][0] | 1, self.RET)

    def tick(self, fin):
        mu = self.mu
        mu.mem_write(self.a_in, bytes(fin))
        self._counting = bool(self.cycle_every) and self.n % self.cycle_every == 0
        self._acc = 0
        self._call('fc_arm_tick')
        if self._counting:
            self.cycles = self._acc
            self.cycle_log.append(self._acc)
        self._counting = False
        self.n += 1
        self.out = FcOut.from_buffer_copy(bytes(mu.mem_read(self.a_out, C.sizeof(FcOut))))
        return self.out

    def stack_high_water(self):
        mem = bytes(self.mu.mem_read(self.stack_paint_from, 0x2000))
        i = 0
        while i < len(mem) and mem[i] == 0xA5:
            i += 1
        return 0x2000 - i


class LockstepFC:
    """Native and ARM builds side by side on identical inputs; any output difference raises."""

    def __init__(self, cfg=None, cycle_every=0):
        self.arm = ArmFC(cfg, cycle_every)
        self.nat = NativeFC(self.arm.cfg)
        self.cfg = self.arm.cfg
        self.n = 0

    def default_config(self):
        return self.cfg

    @property
    def cycles(self):
        return self.arm.cycles

    def tick(self, fin):
        a = bytes(self.arm.tick(fin))
        b = bytes(self.nat.tick(fin))
        if a != b:
            fa, fb = FcOut.from_buffer_copy(a), FcOut.from_buffer_copy(b)
            diff = [f for f, _ in FcOut._fields_ if bytes(getattr(fa, f)) != bytes(getattr(fb, f))
                    if hasattr(getattr(fa, f), '__len__')] + \
                   [f for f, _ in FcOut._fields_ if not hasattr(getattr(fa, f), '__len__') and getattr(fa, f) != getattr(fb, f)]
            raise AssertionError(f'ARM and native outputs differ at tick {self.n}: {diff}')
        self.n += 1
        self.out = self.arm.out
        return self.out
