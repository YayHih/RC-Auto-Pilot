"""Interactive viewer:  python viewer.py [--seed N] [--selftest]

Buttons (or keys): TAKEOFF [t]  HOVER [h]  LOOP [l]  LAND [a]  RESET [r]  FOLLOW [f]  SPEED [s]
Everything the autopilot sees goes through the noisy GPS/compass/IMU/baro models.
"""
import sys
import time
import numpy as np
import matplotlib
if '--selftest' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from rcsim.world import Sim, DT
from rcsim.autopilot import ORBIT_C, ORBIT_R
from rcsim.mathutil import R2euler

SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 3
SCALE = 7.0            # draw the aircraft this much bigger than life

# aircraft glyph in body frame (x fwd, y right, z down)
BODY = {
    'fuse': np.array([[0.5, 0, 0], [-0.5, 0, 0]]),
    'wing': np.array([[0.05, -0.5, 0], [0.05, 0.5, 0]]),
    'tail': np.array([[-0.45, -0.17, 0], [-0.45, 0.17, 0]]),
    'fin': np.array([[-0.45, 0, 0], [-0.45, 0, -0.2]]),
    'nose': np.array([[0.5, 0, 0], [0.35, 0, 0]]),
}


class App:
    def __init__(self):
        self.sim = Sim(seed=SEED)
        self.speed = 1.0
        self.follow = True
        self.last = time.time()
        self.fig = plt.figure(figsize=(13, 7.5))
        self.fig.canvas.manager.set_window_title('RC flight sim - noisy GPS/compass autopilot') \
            if self.fig.canvas.manager else None
        self.ax3 = self.fig.add_axes([0.01, 0.14, 0.58, 0.84], projection='3d')
        self.axm = self.fig.add_axes([0.63, 0.50, 0.35, 0.46])
        self.axt = self.fig.add_axes([0.63, 0.14, 0.35, 0.30])
        self.axt.axis('off')
        self._static()
        self.lines = {k: self.ax3.plot([], [], [], lw=3 if k != 'nose' else 4,
                                        color='crimson' if k == 'nose' else 'k')[0] for k in BODY}
        self.trail, = self.ax3.plot([], [], [], lw=1, color='tab:blue')
        self.shadow, = self.ax3.plot([], [], [], 'o', ms=4, color='gray')
        self.m_trail, = self.axm.plot([], [], lw=1, color='tab:blue')
        self.m_est, = self.axm.plot([], [], lw=0.8, color='tab:orange', alpha=0.8)
        self.m_dot, = self.axm.plot([], [], 'o', color='crimson')
        self.text = self.axt.text(0, 1, '', va='top', family='monospace', fontsize=9)
        self._buttons()
        self.fig.canvas.mpl_connect('key_press_event', self._key)
        self.trail_pts = []
        self.est_pts = []

    def _static(self):
        a = self.ax3
        a.set_xlabel('East'); a.set_ylabel('North'); a.set_zlabel('Alt')
        # runway
        rw = np.array([[-4, -30], [4, -30], [4, 90], [-4, 90], [-4, -30]])
        a.plot(rw[:, 0], rw[:, 1], np.zeros(5), color='dimgray', lw=2)
        th = np.linspace(0, 2 * np.pi, 100)
        a.plot(ORBIT_C[1] + ORBIT_R * np.cos(th), ORBIT_C[0] + ORBIT_R * np.sin(th),
               np.full(100, 28.0), ':', color='green', lw=1)
        m = self.axm
        m.plot(rw[:, 0], rw[:, 1], color='dimgray', lw=3)
        m.plot(ORBIT_C[1] + ORBIT_R * np.cos(th), ORBIT_C[0] + ORBIT_R * np.sin(th), ':', color='green')
        m.set_aspect('equal'); m.grid(alpha=.3); m.set_title('Top view  (blue = true, orange = GPS/IMU estimate)', fontsize=9)
        m.set_xlim(-130, 230); m.set_ylim(-240, 170)

    def _buttons(self):
        specs = [('TAKEOFF', 'takeoff', 'seagreen'), ('HOVER', 'hover', 'steelblue'),
                 ('LOOP', 'loop', 'darkorange'), ('LAND', 'land', 'firebrick')]
        self.btns = []
        x = 0.04
        for label, cmd, col in specs:
            ax = self.fig.add_axes([x, 0.02, 0.10, 0.07])
            b = Button(ax, label, color=col, hovercolor='gold')
            b.label.set_color('white'); b.label.set_fontweight('bold')
            b.on_clicked(lambda e, c=cmd: self.sim.press(c))
            self.btns.append(b); x += 0.115
        for label, fn in [('RESET', self.reset), ('FOLLOW', self.toggle_follow), ('SPEED x1', self.cycle_speed)]:
            ax = self.fig.add_axes([x, 0.02, 0.09, 0.07])
            b = Button(ax, label)
            b.on_clicked(lambda e, f=fn: f())
            self.btns.append(b); x += 0.10
        self.speed_btn = self.btns[-1]

    def _key(self, ev):
        k = ev.key
        cmds = {'t': 'takeoff', 'h': 'hover', 'l': 'loop', 'a': 'land'}
        if k in cmds:
            self.sim.press(cmds[k])
        elif k == 'r':
            self.reset()
        elif k == 'f':
            self.toggle_follow()
        elif k == 's':
            self.cycle_speed()

    def reset(self):
        self.sim = Sim(seed=int(time.time()) % 1000)
        self.trail_pts, self.est_pts = [], []

    def toggle_follow(self):
        self.follow = not self.follow

    def cycle_speed(self):
        self.speed = {1.0: 2.0, 2.0: 4.0, 4.0: 0.5, 0.5: 1.0}[self.speed]
        self.speed_btn.label.set_text(f'SPEED x{self.speed:g}')

    def advance(self, wall_dt):
        n = int(min(wall_dt, 0.1) * self.speed / DT)
        s = self.sim
        for _ in range(n):
            if s.af.crashed:
                break
            s.step()

    def draw(self):
        s = self.sim
        x = s.x
        P = np.array([x[1], x[0], -x[2]])        # plot as (East, North, Up)
        from rcsim.mathutil import q2R
        R = q2R(x[6:10])
        for k, pts in BODY.items():
            w = (R @ (pts * SCALE).T).T + x[0:3]
            self.lines[k].set_data_3d(w[:, 1], w[:, 0], -w[:, 2])
        self.trail_pts.append(P)
        self.est_pts.append([s.est.pos[1], s.est.pos[0]])
        if len(self.trail_pts) > 2500:
            self.trail_pts = self.trail_pts[-2500:]
            self.est_pts = self.est_pts[-2500:]
        T = np.array(self.trail_pts)
        self.trail.set_data_3d(T[:, 0], T[:, 1], T[:, 2])
        self.shadow.set_data_3d([P[0]], [P[1]], [0])
        self.m_trail.set_data(T[:, 0], T[:, 1])
        E = np.array(self.est_pts)
        self.m_est.set_data(E[:, 0], E[:, 1])
        self.m_dot.set_data([P[0]], [P[1]])
        a = self.ax3
        if self.follow:
            r = 45
            a.set_xlim(P[0] - r, P[0] + r); a.set_ylim(P[1] - r, P[1] + r); a.set_zlim(max(P[2] - r, 0), max(P[2] + r, 2 * r * 0.5))
        else:
            a.set_xlim(-130, 230); a.set_ylim(-240, 170); a.set_zlim(0, 90)
        a.set_box_aspect((1, 1, 0.5) if not self.follow else (1, 1, 0.8))
        ap, est = s.ap, s.est
        ty, tp, tr = R2euler(R)
        ey, ep, er = est.euler()
        V = np.linalg.norm(x[3:6])
        txt = (f"STATE   {ap.state:10s} goal: {ap.goal}\n"
               f"time    {s.t:6.1f} s     speed {V:5.1f} m/s\n"
               f"alt     true {-x[2]:6.1f}  est {est.alt:6.1f} m\n"
               f"pos err {np.linalg.norm(x[0:3] - est.pos):5.1f} m   (GPS 10 Hz, ~1.5 m CEP)\n"
               f"pitch   true {np.degrees(tp):6.1f}  est {np.degrees(ep):6.1f}\n"
               f"roll    true {np.degrees(tr):6.1f}  est {np.degrees(er):6.1f}\n"
               f"yaw     true {np.degrees(ty):6.1f}  est {np.degrees(ey):6.1f}\n"
               f"throttle {s.mot:4.2f}  ail {s.servo[0]:+.2f}  ele {s.servo[1]:+.2f}  rud {s.servo[2]:+.2f}\n")
        if s.af.crashed:
            txt += f"\nCRASHED: {s.af.crash_reason}\n(press RESET)"
        self.text.set_text(txt)

    def frame(self, _):
        now = time.time()
        self.advance(now - self.last)
        self.last = now
        self.draw()
        return []


def main():
    app = App()
    if '--selftest' in sys.argv:
        app.sim.press('takeoff'); app.sim.press('loop')
        for i in range(40):
            app.advance(0.1 * 4)
            app.draw()
        app.fig.savefig('viewer_selftest.png', dpi=80)
        print('selftest ok', app.sim.ap.state, round(app.sim.t, 1), 'crashed', app.sim.af.crashed)
        return
    app.anim = FuncAnimation(app.fig, app.frame, interval=40, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    main()
