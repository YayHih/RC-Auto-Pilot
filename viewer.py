"""Interactive viewer (pygame):  python viewer.py [--seed N] [--wind M/S] [--wind-from DEG] [--turb X] [--no-random]
                                                  [--fc native|arm] [--plane ail=0.6,ele=1.4,rud=1,mass=1.2,thrust=1]
                                                  [--morph S]   (random in-flight changes every ~S s: control-surface
                                                                 size, wing chord, drag plate - drawn on the model)
                                                  [--screenshots DIR]   (headless: render each camera to DIR, report timing)
The aeroplane is flown by the C++ flight controller in firmware/RCAutopilot/src/fc - the same code that
runs on the Heltec (--fc arm runs its Cortex-M0+ machine code in an emulator; slower than real time).

Auto buttons  : TAKEOFF [1/t]  HOVER [2/h]  LOOP [3/l]  LAND [4/g]
Manual flying : MANUAL [m] takes the stick, ASSIST [n] toggles stabilised (angle) / direct (raw surfaces) mode
                arrows  elevator + aileron  (down arrow = pull back = climb, up = push = dive)
                a / d   rudder (nose-wheel steering on the ground)       w / s   throttle up / down
                z       throttle cut                                     x       full throttle
                or drag in the CONTROLS panel with the mouse (stick square, throttle bar, rudder bar)
Wind          : sliders at the bottom (speed, direction it blows FROM, turbulence, gusts)
                keys [ ] speed   , . direction   - = turbulence   u i gusts
View          : VIEW [c] cycles chase / observer / FPV / far-chase / tower,  RESET [r],  SPEED [v],  pause [p],
                Q cycles render quality (default: auto, adapts to hold a high frame rate),  esc quits
Everything the autopilot sees goes through the noisy GPS/compass/IMU/baro models; every run has random
build tolerances, wind and gusts (seeded; --no-random gives the nominal airframe).
"""
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HEADLESS = '--screenshots' in sys.argv
if HEADLESS:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame
import pygame.gfxdraw as gfx

from rcsim.world import Sim, DT
from rcsim.mathutil import R2euler, q2R
from rcsim.render import FrameBuilder, RUNWAY_X, RUNWAY_HALF_W


def _arg(name, default=None, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SEED = _arg('--seed', 3, int)
WIND = _arg('--wind', None)
WIND_FROM = _arg('--wind-from', None)
TURB = _arg('--turb', 1.0)
RANDOMIZE = 0.0 if '--no-random' in sys.argv else 1.0
FC_KIND = _arg('--fc', 'native', str)
_PL = _arg('--plane', None, str)
PLANE = {k: float(v) for k, v in (kv.split('=') for kv in _PL.split(','))} if _PL else None
MORPH = _arg('--morph', 0.0)
FENCE_M = 15.0                         # cfg.fence_m: manoeuvres stay east of this (cfg.fly_side = +1)
ORBIT_R = 60.0
ORBIT_C = np.array([90.0, FENCE_M + ORBIT_R + 5.0])   # loiter circle (autopilot.cpp, runway heading north)

WIN = (1440, 860)
VIEW = pygame.Rect(10, 10, 1010, 640)
BG = (20, 23, 28)
PANEL = (27, 31, 38)
EDGE = (58, 65, 77)
FG = (216, 221, 230)
DIM = (150, 158, 172)
QUALITY = [(280, 179), (384, 246), (480, 307), (640, 410), (800, 512)]      # backdrop resolutions


def _ints(pts):
    if isinstance(pts, list):
        return pts
    return [(int(x), int(y)) for x, y in pts]


class Button:
    def __init__(self, rect, label, color, cb):
        self.rect, self.label, self.color, self.cb = pygame.Rect(rect), label, color, cb

    def draw(self, surf, font, hover, text=None):
        c = tuple(min(255, v + 40) for v in self.color) if hover else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=6)
        t = font.render(text or self.label, True, (255, 255, 255))
        surf.blit(t, t.get_rect(center=self.rect.center))


class Slider:
    def __init__(self, rect, label, lo, hi, key, fmt='%.1f'):
        self.rect, self.label, self.lo, self.hi, self.key, self.fmt = pygame.Rect(rect), label, lo, hi, key, fmt
        self.val = lo

    def frac(self):
        return (self.val - self.lo) / (self.hi - self.lo)

    def set_from_x(self, x):
        f = np.clip((x - self.rect.x) / self.rect.w, 0, 1)
        self.val = float(self.lo + f * (self.hi - self.lo))

    def draw(self, surf, font):
        t = font.render(f'{self.label}   {self.fmt % self.val}', True, FG)
        surf.blit(t, (self.rect.x, self.rect.y - 20))
        pygame.draw.rect(surf, (38, 44, 54), self.rect, border_radius=4)
        fill = pygame.Rect(self.rect.x, self.rect.y, int(self.rect.w * self.frac()), self.rect.h)
        pygame.draw.rect(surf, (82, 214, 255), fill, border_radius=4)
        pygame.draw.circle(surf, (255, 255, 255), (self.rect.x + int(self.rect.w * self.frac()), self.rect.centery), 9)


class App:
    MAP = pygame.Rect(1030, 10, 400, 290)
    ATT = pygame.Rect(1030, 312, 170, 170)
    CTL = pygame.Rect(1210, 312, 220, 190)
    TXT = pygame.Rect(1030, 512, 400, 140)
    STICK = pygame.Rect(1224, 345, 96, 96)
    THR = pygame.Rect(1334, 345, 24, 96)
    RUD = pygame.Rect(1224, 468, 96, 18)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WIN, pygame.DOUBLEBUF)
        pygame.display.set_caption('RC flight sim - noisy-sensor autopilot / manual')
        names = 'menlo,monaco,consolas,dejavusansmono,monospace'
        self.font = pygame.font.SysFont(names, 14)
        self.fontb = pygame.font.SysFont(names, 15, bold=True)
        self.fontbig = pygame.font.SysFont(names, 40, bold=True)
        self.fonts = pygame.font.SysFont(names, 12)
        self.clock = pygame.time.Clock()
        self.q_idx = 1
        self.auto_q = True
        self.fb = FrameBuilder(*QUALITY[self.q_idx])
        self.seed = SEED
        self.speed = 1.0
        self.paused = False
        self.acc = 0.0
        self.mouse = None
        self.drag_slider = None
        self.wind_set = dict(speed=None, from_deg=None, turb=None, gusts=None)
        self.sim = self._new_sim(self.seed)
        self.trail, self.est_trail = [], []
        self.t_render = 16.0
        self.dt, self.fps = 0.016, 60.0
        self.view_surf = pygame.Surface(VIEW.size)
        self._bd = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self._build_ui()
        self._build_static()
        self._sync_sliders()

    # ------------------------------------------------------------------ sim control
    def _new_sim(self, seed):
        return Sim(seed=seed, wind=WIND, wind_from=WIND_FROM, turb=TURB, randomize=RANDOMIZE, fc=FC_KIND, plane=PLANE,
                   morph=MORPH)

    def reset(self):
        was_manual, was_assist = self.sim.manual, self.sim.assist
        self.seed = int(time.time() * 1000) % 100000
        self.sim = self._new_sim(self.seed)
        for k, v in self.wind_set.items():
            if v is not None:
                self.sim.atm.set_wind(**{k: v})
        self.sim.assist = was_assist
        if was_manual:
            self.sim.set_manual(True)
        self.trail, self.est_trail = [], []
        self._sync_sliders()

    def press(self, cmd):
        if self.sim.manual:
            self.sim.set_manual(False)
        self.sim.press(cmd)

    def toggle_manual(self):
        self.sim.set_manual(not self.sim.manual)

    def toggle_assist(self):
        self.sim.assist = not self.sim.assist

    def cycle_speed(self):
        self.speed = {1.0: 2.0, 2.0: 4.0, 4.0: 0.5, 0.5: 1.0}[self.speed]

    def cycle_quality(self):
        self.auto_q = False
        self.q_idx = (self.q_idx + 1) % len(QUALITY)
        self.fb.W, self.fb.H = QUALITY[self.q_idx]

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        specs = [('TAKEOFF', lambda: self.press('takeoff'), (46, 139, 87)), ('HOVER', lambda: self.press('hover'), (59, 120, 168)),
                 ('LOOP', lambda: self.press('loop'), (217, 130, 43)), ('LAND', lambda: self.press('land'), (184, 67, 58)),
                 ('MANUAL', self.toggle_manual, (123, 82, 184)), ('ASSIST', self.toggle_assist, (75, 138, 138)),
                 ('RESET', self.reset, (85, 91, 102)), ('VIEW', self.fb.cycle_mode, (85, 91, 102)),
                 ('SPEED', self.cycle_speed, (85, 91, 102))]
        self.buttons = {}
        x = 10
        for label, fn, col in specs:
            self.buttons[label] = Button((x, 662, 148, 46), label, col, fn)
            x += 156
        self.sliders = {k: Slider((10 + i * 358, 750, 330, 12), lab, lo, hi, k)
                        for i, (k, lab, lo, hi) in enumerate((('speed', 'wind m/s', 0, 12), ('from_deg', 'wind from deg', 0, 360),
                                                              ('turb', 'turbulence', 0, 3), ('gusts', 'gusts', 0, 3)))}
        self.sliders['from_deg'].fmt = '%.0f'

    def _sync_sliders(self):
        a = self.sim.atm
        for k, v in (('speed', a.W20), ('from_deg', a.from_deg), ('turb', a.turb), ('gusts', a.gusts)):
            self.sliders[k].val = float(v)

    def _slider_changed(self, k):
        v = self.sliders[k].val
        self.wind_set[k] = v
        self.sim.atm.set_wind(**{k: v})

    def _nudge(self, key, d, wrap=False):
        sl = self.sliders[key]
        sl.val = (sl.val + d) % 360.0 if wrap else min(sl.hi, max(sl.lo, sl.val + d))
        self._slider_changed(key)

    def _build_static(self):
        m = pygame.Surface(self.MAP.size)
        m.fill((24, 30, 24))
        xl, yl = (-170, 250), (-250, 170)                           # east, north limits
        sx = lambda e: int((e - xl[0]) / (xl[1] - xl[0]) * (self.MAP.w - 1))
        sy = lambda n: int((yl[1] - n) / (yl[1] - yl[0]) * (self.MAP.h - 1))
        self._msx, self._msy = sx, sy
        for e in range(-150, 251, 50):
            pygame.draw.line(m, (40, 48, 40), (sx(e), 0), (sx(e), self.MAP.h))
        for n in range(-250, 171, 50):
            pygame.draw.line(m, (40, 48, 40), (0, sy(n)), (self.MAP.w, sy(n)))
        pygame.draw.polygon(m, (95, 100, 108), [(sx(-RUNWAY_HALF_W), sy(RUNWAY_X[0])), (sx(RUNWAY_HALF_W), sy(RUNWAY_X[0])),
                                                (sx(RUNWAY_HALF_W), sy(RUNWAY_X[1])), (sx(-RUNWAY_HALF_W), sy(RUNWAY_X[1]))])
        pygame.draw.circle(m, (60, 140, 70), (sx(ORBIT_C[1]), sy(ORBIT_C[0])), int(ORBIT_R / (xl[1] - xl[0]) * self.MAP.w), 1)
        # safety: the pilot stands west of the runway; everything is flown east of the red fence line
        for n in range(-250, 171, 12):
            pygame.draw.line(m, (190, 70, 60), (sx(FENCE_M), sy(n)), (sx(FENCE_M), sy(n + 6)))
        pygame.draw.circle(m, (230, 200, 90), (sx(-15), sy(0)), 4)
        m.blit(self.fonts.render('pilot', True, (230, 200, 90)), (sx(-15) - 38, sy(0) - 7))
        pygame.draw.rect(m, EDGE, m.get_rect(), 1)
        m.blit(self.fonts.render('top view  (blue = true, orange = GPS/IMU estimate)', True, DIM), (6, 4))
        self.map_bg = m
        fr = pygame.Surface(self.ATT.size, pygame.SRCALPHA)
        fr.fill((*PANEL, 255))
        c = (self.ATT.w // 2, self.ATT.h // 2)
        pygame.draw.circle(fr, (0, 0, 0, 0), c, 80)
        pygame.draw.circle(fr, (140, 150, 165, 255), c, 80, 3)
        self.att_frame = fr
        self.att_surf = pygame.Surface(self.ATT.size)

    # ------------------------------------------------------------------ input
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                return False
            once = {pygame.K_1: lambda: self.press('takeoff'), pygame.K_t: lambda: self.press('takeoff'),
                    pygame.K_2: lambda: self.press('hover'), pygame.K_h: lambda: self.press('hover'),
                    pygame.K_3: lambda: self.press('loop'), pygame.K_l: lambda: self.press('loop'),
                    pygame.K_4: lambda: self.press('land'), pygame.K_g: lambda: self.press('land'),
                    pygame.K_m: self.toggle_manual, pygame.K_n: self.toggle_assist, pygame.K_c: self.fb.cycle_mode,
                    pygame.K_r: self.reset, pygame.K_v: self.cycle_speed, pygame.K_q: self.cycle_quality,
                    pygame.K_p: lambda: setattr(self, 'paused', not self.paused),
                    pygame.K_z: lambda: self.sim.stick.update(thr=0.0), pygame.K_x: lambda: self.sim.stick.update(thr=1.0),
                    pygame.K_LEFTBRACKET: lambda: self._nudge('speed', -0.5), pygame.K_RIGHTBRACKET: lambda: self._nudge('speed', 0.5),
                    pygame.K_COMMA: lambda: self._nudge('from_deg', -15, True), pygame.K_PERIOD: lambda: self._nudge('from_deg', 15, True),
                    pygame.K_MINUS: lambda: self._nudge('turb', -0.25), pygame.K_EQUALS: lambda: self._nudge('turb', 0.25),
                    pygame.K_u: lambda: self._nudge('gusts', -0.25), pygame.K_i: lambda: self._nudge('gusts', 0.25)}
            if k in once:
                once[k]()
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            pos = ev.pos
            for b in self.buttons.values():
                if b.rect.collidepoint(pos):
                    b.cb()
                    return True
            for k, sl in self.sliders.items():
                if sl.rect.inflate(20, 24).collidepoint(pos):
                    self.drag_slider = k
                    sl.set_from_x(pos[0])
                    self._slider_changed(k)
                    return True
            hit = self._ctl_hit(pos)
            if hit:
                if not self.sim.manual:
                    self.sim.set_manual(True)
                self.mouse = hit
                self._apply_mouse(pos)
        elif ev.type == pygame.MOUSEMOTION:
            if self.drag_slider:
                self.sliders[self.drag_slider].set_from_x(ev.pos[0])
                self._slider_changed(self.drag_slider)
            elif self.mouse:
                self._apply_mouse(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag_slider = None
            self.mouse = None
        return True

    def _ctl_hit(self, pos):
        if self.STICK.inflate(8, 8).collidepoint(pos):
            return 'stick'
        if self.THR.inflate(30, 10).collidepoint(pos):
            return 'thr'
        if self.RUD.inflate(8, 14).collidepoint(pos):
            return 'rud'
        return None

    def _apply_mouse(self, pos):
        st = self.sim.stick
        if self.mouse == 'stick':
            st['da'] = float(np.clip((pos[0] - self.STICK.centerx) / (self.STICK.w / 2), -1, 1))
            st['de'] = float(np.clip((pos[1] - self.STICK.centery) / (self.STICK.h / 2), -1, 1))   # screen down = pull = +
        elif self.mouse == 'thr':
            st['thr'] = float(np.clip((self.THR.bottom - pos[1]) / self.THR.h, 0, 1))
        elif self.mouse == 'rud':
            st['dr'] = float(np.clip((pos[0] - self.RUD.centerx) / (self.RUD.w / 2), -1, 1))

    def update_stick(self, dt):
        s = self.sim
        if not s.manual or self.mouse:
            return
        k = pygame.key.get_pressed()
        st = s.stick
        tgt = dict(da=float(k[pygame.K_RIGHT]) - float(k[pygame.K_LEFT]),
                   de=float(k[pygame.K_DOWN]) - float(k[pygame.K_UP]),
                   dr=float(k[pygame.K_d]) - float(k[pygame.K_a]))
        for key, tg in tgt.items():
            rate = 6.0 if tg != 0 else 9.0
            st[key] += float(np.clip(tg - st[key], -rate * dt, rate * dt))
        if k[pygame.K_w]:
            st['thr'] = min(1.0, st['thr'] + 0.7 * dt)
        if k[pygame.K_s]:
            st['thr'] = max(0.0, st['thr'] - 0.7 * dt)

    # ------------------------------------------------------------------ drawing
    def _text(self, surf, txt, pos, font=None, color=(255, 255, 255), shadow=True, anchor='topleft'):
        font = font or self.font
        t = font.render(txt, True, color)
        r = t.get_rect(**{anchor: pos})
        if shadow:
            sh = font.render(txt, True, (0, 0, 0))
            for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                surf.blit(sh, r.move(dx, dy))
        surf.blit(t, r)

    def draw_view(self):
        """sky/ground ray-cast runs on a worker thread (numpy releases the GIL) while this thread builds and
        draws the polygons, the panels and steps the physics"""
        s = self.sim
        fb = self.fb
        fb.setup_camera(s, min(self.dt, 0.1))
        job = self.pool.submit(fb.backdrop, s)
        shadows, items = fb.items(s, VIEW.w, VIEW.h)
        self.draw_map()
        self.draw_attitude()
        self.draw_controls()
        self.draw_text()
        self.draw_ui()
        img = job.result()
        if self._bd is None or self._bd.get_size() != (fb.W, fb.H):
            self._bd = pygame.Surface((fb.W, fb.H), 0, 32)
        pygame.surfarray.blit_array(self._bd, img.swapaxes(0, 1))
        pygame.transform.smoothscale(self._bd, VIEW.size, self.view_surf)
        v = self.view_surf
        for pts, col, kind in shadows:
            p = _ints(pts)
            if len(p) >= 3:
                gfx.filled_polygon(v, p, col)
        for pts, col, kind in items:
            p = _ints(pts)
            if len(p) < 3:
                continue
            gfx.filled_polygon(v, p, col)
            if col[3] > 200 and kind == 'p':
                gfx.aapolygon(v, p, (int(col[0] * 0.55), int(col[1] * 0.55), int(col[2] * 0.55), 255))
        self._hud(v)
        self.screen.blit(v, VIEW.topleft)
        pygame.draw.rect(self.screen, EDGE, VIEW, 1)

    def _hud(self, v):
        s, x = self.sim, self.sim.x
        V = float(np.linalg.norm(x[3:6]))
        if s.manual:
            mode = 'MANUAL  ' + ('ASSIST (angle mode)' if s.assist else 'DIRECT (raw surfaces)')
        else:
            mode = f'AUTO    {s.ap.state}' + (f' -> {s.ap.goal}' if s.ap.goal else '')
        panel = pygame.Surface((340, 50), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 105))
        v.blit(panel, (10, 10))
        self._text(v, mode, (18, 14), self.fontb)
        self._text(v, f'{V * 3.6:5.0f} km/h   alt {-x[2]:5.1f} m   vs {-x[5]:+5.1f} m/s', (18, 34))
        info = f'view: {self.fb.mode}   seed {self.seed}   sim x{self.speed:g}   {self.fps:3.0f} fps   {self.fb.W}x{self.fb.H}'
        self._text(v, info + ('   PAUSED' if self.paused else ''), (12, VIEW.h - 22), self.fonts)
        w = s.atm.mean(6.0)
        wsp = float(np.hypot(w[0], w[1]))
        cx, cy, r = VIEW.w - 70, 72, 46
        ps = pygame.Surface((2 * r + 4, 2 * r + 4), pygame.SRCALPHA)
        pygame.draw.circle(ps, (0, 0, 0, 115), (r + 2, r + 2), r)
        pygame.draw.circle(ps, (255, 255, 255, 190), (r + 2, r + 2), r, 1)
        v.blit(ps, (cx - r - 2, cy - r - 2))
        yaw = self.fb.cam_yaw if self.fb.mode in ('chase', 'far-chase') else \
            (R2euler(q2R(x[6:10]))[0] if self.fb.mode == 'fpv' else 0.0)
        rel = np.arctan2(w[1], w[0]) - yaw
        L = r * (0.45 + 0.5 * min(wsp / 8.0, 1.0))
        d = np.array([np.sin(rel), -np.cos(rel)])
        tail, head = np.array([cx, cy]) - d * L * 0.5, np.array([cx, cy]) + d * L * 0.5
        pygame.draw.line(v, (82, 214, 255), tail, head, 4)
        n = np.array([-d[1], d[0]])
        pygame.draw.polygon(v, (82, 214, 255), [head + d * 9, head - d * 4 + n * 7, head - d * 4 - n * 7])
        self._text(v, 'WIND', (cx, cy - r - 4), self.fonts, (190, 234, 255), anchor='midbottom')
        self._text(v, f'{wsp:3.1f} m/s from {s.atm.from_deg:3.0f}', (cx, cy + r + 4), self.fonts, anchor='midtop')
        self._text(v, f'turb {s.atm.turb:.1f} gust {s.atm.gusts:.1f}', (cx, cy + r + 18), self.fonts, anchor='midtop')
        if s.af.crashed:
            self._text(v, 'CRASHED', (VIEW.w // 2, 250), self.fontbig, (255, 90, 74), anchor='midtop')
            self._text(v, s.af.crash_reason[:60], (VIEW.w // 2, 298), self.font, (255, 200, 190), anchor='midtop')
            self._text(v, 'press R to reset', (VIEW.w // 2, 320), self.font, anchor='midtop')

    def draw_map(self):
        s = self.sim
        m = self.map_bg.copy()
        T, E = self.trail[-900:][::2], self.est_trail[-900:][::2]
        if len(E) > 1:
            pygame.draw.lines(m, (255, 159, 67), False, [(self._msx(e), self._msy(n)) for n, e in E], 1)
        if len(T) > 1:
            pygame.draw.lines(m, (74, 163, 255), False, [(self._msx(e), self._msy(n)) for n, e in T], 2)
        x = s.x
        px, py = self._msx(x[1]), self._msy(x[0])
        R = q2R(x[6:10])
        h = np.array([R[1, 0], -R[0, 0]])
        h = h / (np.linalg.norm(h) + 1e-9)
        n = np.array([-h[1], h[0]])
        pygame.draw.polygon(m, (255, 77, 77), [(px + h[0] * 9, py + h[1] * 9), (px - h[0] * 5 + n[0] * 5, py - h[1] * 5 + n[1] * 5),
                                               (px - h[0] * 5 - n[0] * 5, py - h[1] * 5 - n[1] * 5)])
        w = s.atm.mean(6.0)
        c0 = (60, 40)
        pygame.draw.line(m, (82, 214, 255), c0, (c0[0] + w[1] * 7, c0[1] - w[0] * 7), 2)
        pygame.draw.circle(m, (82, 214, 255), c0, 3)
        self.screen.blit(m, self.MAP.topleft)

    def draw_attitude(self):
        _, pitch, roll = R2euler(q2R(self.sim.x[6:10]))
        a = self.att_surf
        a.fill((61, 134, 214))
        c = np.array([self.ATT.w / 2, self.ATT.h / 2])
        k = 80.0 / 28.0
        yh = np.degrees(pitch) * k
        cr, sr = np.cos(roll), np.sin(roll)

        def rot(px, py):
            return (c[0] + cr * px - sr * py, c[1] - (sr * px + cr * py))
        big = 400
        pygame.draw.polygon(a, (138, 90, 43), [rot(-big, -yh), rot(big, -yh), rot(big, -yh - big), rot(-big, -yh - big)])
        pygame.draw.line(a, (255, 255, 255), rot(-big, -yh), rot(big, -yh), 2)
        for deg in (-30, -20, -10, 10, 20, 30):
            half = 14 if deg % 20 else 24
            pygame.draw.line(a, (255, 255, 255), rot(-half, -yh + deg * k), rot(half, -yh + deg * k), 1)
        a.blit(self.att_frame, (0, 0))
        cx, cy = int(c[0]), int(c[1])
        pygame.draw.line(a, (255, 210, 63), (cx - 48, cy), (cx - 16, cy), 3)
        pygame.draw.line(a, (255, 210, 63), (cx + 16, cy), (cx + 48, cy), 3)
        pygame.draw.circle(a, (255, 210, 63), (cx, cy), 3)
        self.screen.blit(a, self.ATT.topleft)
        self._text(self.screen, 'attitude', (self.ATT.x + 4, self.ATT.y), self.fonts, DIM, shadow=False)

    def draw_controls(self):
        s = self.sim
        scr = self.screen
        pygame.draw.rect(scr, PANEL, self.CTL)
        pygame.draw.rect(scr, EDGE, self.CTL, 1)
        self._text(scr, 'controls (drag to fly)', (self.CTL.x + 6, self.CTL.y + 2), self.fonts, DIM, shadow=False)
        act = s.out_now
        cmd = dict(s.stick) if s.manual else dict(da=act['da'], de=act['de'], dr=act['dr'], thr=act['thr'])
        sk = self.STICK
        pygame.draw.rect(scr, (29, 35, 44), sk)
        pygame.draw.rect(scr, (106, 116, 132), sk, 1)
        pygame.draw.line(scr, (57, 65, 80), (sk.centerx, sk.y), (sk.centerx, sk.bottom))
        pygame.draw.line(scr, (57, 65, 80), (sk.x, sk.centery), (sk.right, sk.centery))
        self._text(scr, 'AIL/ELEV', (sk.centerx, sk.y - 3), self.fonts, DIM, shadow=False, anchor='midbottom')
        if s.manual and s.assist:
            pygame.draw.circle(scr, (255, 122, 89), (int(sk.centerx + act['da'] * sk.w / 2), int(sk.centery + act['de'] * sk.h / 2)), 10, 2)
        pygame.draw.circle(scr, (255, 210, 63) if s.manual else (110, 119, 135),
                           (int(sk.centerx + cmd['da'] * sk.w / 2), int(sk.centery + cmd['de'] * sk.h / 2)), 6)
        th = self.THR
        pygame.draw.rect(scr, (29, 35, 44), th)
        hh = int(th.h * float(np.clip(cmd['thr'], 0, 1)))
        pygame.draw.rect(scr, (98, 210, 111), (th.x, th.bottom - hh, th.w, hh))
        pygame.draw.rect(scr, (106, 116, 132), th, 1)
        self._text(scr, 'THR', (th.centerx, th.y - 3), self.fonts, DIM, shadow=False, anchor='midbottom')
        ru = self.RUD
        pygame.draw.rect(scr, (29, 35, 44), ru)
        pygame.draw.rect(scr, (106, 116, 132), ru, 1)
        pygame.draw.line(scr, (57, 65, 80), (ru.centerx, ru.y), (ru.centerx, ru.bottom))
        pygame.draw.rect(scr, (255, 210, 63), (int(ru.centerx + cmd['dr'] * ru.w / 2) - 4, ru.y + 2, 8, ru.h - 4))
        self._text(scr, 'RUDDER', (ru.centerx, ru.bottom + 2), self.fonts, DIM, shadow=False, anchor='midtop')
        sv = s.servo
        for i, (lbl, v) in enumerate(zip(('L', 'R', 'E', 'Ru'), (-sv[0], sv[0], sv[1], sv[2]))):   # TE up = +
            bar = pygame.Rect(1374 + i * 14, sk.y, 12, sk.h)
            pygame.draw.rect(scr, (29, 35, 44), bar)
            pygame.draw.rect(scr, (106, 116, 132), bar, 1)
            v = float(np.clip(v, -1, 1))
            h = int(abs(v) * sk.h / 2)
            if v >= 0:
                pygame.draw.rect(scr, (255, 169, 77), (bar.x + 1, bar.centery - h, bar.w - 2, h))
            else:
                pygame.draw.rect(scr, (110, 193, 255), (bar.x + 1, bar.centery, bar.w - 2, h))
            self._text(scr, lbl, (bar.centerx, bar.y - 3), self.fonts, DIM, shadow=False, anchor='midbottom')
        self._text(scr, 'surfaces', (1388, sk.bottom + 3), self.fonts, DIM, shadow=False, anchor='midtop')
        self._text(scr, '(up = TE up)', (1388, sk.bottom + 16), self.fonts, DIM, shadow=False, anchor='midtop')

    def draw_text(self):
        s, x = self.sim, self.sim.x
        est = s.est
        ty, tp, tr = R2euler(q2R(x[6:10]))
        ey, ep, er = est.euler()
        sv = s.servo
        o = s.fout
        we, wt = np.array(o.wind[:]), s.wind_now[:2]
        lines = [f"pos err {np.linalg.norm(x[0:3] - est.pos):5.1f} m   alt true {-x[2]:6.1f} est {est.alt:6.1f}",
                 f"pitch {np.degrees(tp):6.1f} (est {np.degrees(ep):6.1f})  roll {np.degrees(tr):6.1f} (est {np.degrees(er):6.1f})",
                 f"yaw   {np.degrees(ty):6.1f} (est {np.degrees(ey):6.1f})   thr {s.mot:4.2f}  batt {100 * s.af.soc(x[14]):3.0f}%",
                 f"airspeed {np.linalg.norm(x[3:6] - s.wind_now) * 3.6:5.1f} km/h (est {o.airspeed * 3.6:5.1f})  T {s.T / (s.af.m * 9.81):.1f}x wt",
                 f"wind est {np.hypot(*we):4.1f} m/s from {np.degrees(np.arctan2(-we[1], -we[0])) % 360:3.0f}"
                 f"  (true {np.hypot(*wt):4.1f} from {np.degrees(np.arctan2(-wt[1], -wt[0])) % 360:3.0f})",
                 f"learned authority a/e/r x{o.auth_q[0]:.2f}/{o.auth_q[1]:.2f}/{o.auth_q[2]:.2f}  wash x{o.auth_w[1]:.2f}",
                 "plane now: surf x{ail:.2f}/{ele:.2f}/{rud:.2f} wing x{wing:.2f} drag x{drag:.2f}".format(**s.af.morph),
                 f"PCA9685 out {' '.join(str(c) for c in o.pca_off[:5])}   [{s.fc_kind}]"]
        pygame.draw.rect(self.screen, PANEL, self.TXT)
        pygame.draw.rect(self.screen, EDGE, self.TXT, 1)
        for i, ln in enumerate(lines):
            self._text(self.screen, ln, (self.TXT.x + 8, self.TXT.y + 3 + 13 * i), self.fonts, FG, shadow=False)
        hint = (['arrows pitch/roll (down = climb)  a/d rudder', 'w/s throttle  z cut  x full   m = back to auto'] if s.manual else
                ['1 takeoff 2 hover 3 loop 4 land   m manual', 'c view  r reset  q quality  [ ] , . - = u i wind'])
        for i, ln in enumerate(hint):
            self._text(self.screen, ln, (self.TXT.x + 8, self.TXT.y + 110 + 13 * i), self.fonts, DIM, shadow=False)

    def draw_ui(self):
        scr = self.screen
        s = self.sim
        mp = pygame.mouse.get_pos()
        lab = {'MANUAL': 'AUTO' if s.manual else 'MANUAL', 'ASSIST': 'ASSIST ' + ('ON' if s.assist else 'OFF'),
               'SPEED': f'SPEED x{self.speed:g}'}
        for name, b in self.buttons.items():
            b.draw(scr, self.fontb, b.rect.collidepoint(mp), lab.get(name))
        for sl in self.sliders.values():
            sl.draw(scr, self.font)
        self._text(scr, 'wind sliders change the air live; every run draws random build tolerances, wind, wander and gusts from its seed.',
                   (10, 790), self.fonts, DIM, shadow=False)

    def draw_frame(self):
        s = self.sim
        x = s.x
        self.trail.append((x[0], x[1]))
        self.est_trail.append((s.est.pos[0], s.est.pos[1]))
        if len(self.trail) > 4000:
            self.trail, self.est_trail = self.trail[-3000:], self.est_trail[-3000:]
        self.draw_view()

    # ------------------------------------------------------------------ loop
    def step_sim(self):
        if self.paused:
            return
        self.acc += min(self.dt, 0.05) * self.speed
        n = int(self.acc / DT)
        self.acc -= n * DT
        s = self.sim
        for _ in range(n):
            if s.af.crashed:
                break
            s.step()

    def adapt_quality(self):
        if not self.auto_q:
            return
        if self.t_render > 19.0 and self.q_idx > 0:
            self.q_idx -= 1
        elif self.t_render < 12.0 and self.q_idx < 3:
            self.q_idx += 1
        else:
            return
        self.fb.W, self.fb.H = QUALITY[self.q_idx]
        self.t_render = 16.0

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            self.update_stick(self.dt)
            t0 = time.time()
            self.step_sim()
            self.screen.fill(BG)
            self.draw_frame()
            pygame.display.flip()
            self.t_render = 0.9 * self.t_render + 0.1 * (time.time() - t0) * 1000
            self.adapt_quality()
            self.dt = self.clock.tick(60) / 1000.0
            f = self.clock.get_fps()
            if f:
                self.fps = 0.9 * self.fps + 0.1 * f
        pygame.quit()


def screenshots(outdir):
    """headless: fly a takeoff, render every camera, report per-frame cost"""
    os.makedirs(outdir, exist_ok=True)
    app = App()
    app.dt = 0.033
    app.sim.press('takeoff')
    for _ in range(int(14 / 0.033)):
        app.step_sim()
    times = {}
    for mode in app.fb.MODES:
        app.fb.mode = mode
        t0 = time.time()
        n = 8
        for _ in range(n):
            app.step_sim()
            app.screen.fill(BG)
            app.draw_frame()
        times[mode] = (time.time() - t0) / n * 1000
        pygame.image.save(app.screen, os.path.join(outdir, f'{mode}.png'))
    app.sim.set_manual(True)
    app.sim.stick.update(da=0.6, de=-0.4, dr=0.5, thr=0.8)
    app.fb.mode = 'chase'
    app.fb.chase_dist = 2.2
    for _ in range(30):
        app.step_sim()
    app.draw_frame()
    pygame.image.save(app.screen, os.path.join(outdir, 'manual_closeup.png'))
    app.sim.af.set_morph(ail=1.8, ele=1.7, rud=1.7, wing=1.3, drag=3.0)     # the same moment, a changed aeroplane
    app.draw_frame()
    pygame.image.save(app.screen, os.path.join(outdir, 'morph_closeup.png'))
    print('ms/frame incl. sim + UI:', {k: round(v, 1) for k, v in times.items()})


def main():
    if HEADLESS:
        screenshots(sys.argv[sys.argv.index('--screenshots') + 1])
        return
    App().run()


if __name__ == '__main__':
    main()
