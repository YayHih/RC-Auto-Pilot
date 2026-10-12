"""Physics v2: 6-DOF rigid body with a strip-theory aerodynamic model, motor/prop/battery model.

Body axes FRD (x nose, y right, z down), world NED, ground at z = 0, CG at body origin.

Aerodynamics: every lifting surface is a strip (inner/mid/outer wing panels, two tail halves, fin).
Each strip sees its own local airflow = free stream + omega x r (so damping, adverse yaw, roll
coupling all emerge) + propeller slipstream (axial wash and swirl, weighted by how much of the strip
sits in the prop wash), has its own lift/stall/flat-plate/induced-drag curve and flap effectiveness.
Wing strips get ground effect, tail strips see wing downwash, the fuselage is a 3-axis bluff body.

Propulsion: 4S LiPo with internal-resistance sag and state of charge -> ESC duty -> brushless motor
(Kv, R, I0) -> prop with CT/CP(J) -> rotor inertia.  Gyroscopic and reaction torque are applied.

Control inputs (normalised surface positions, -1..1): da>0 right wing down, de>0 nose up, dr>0 nose right.
State x: pos(3) vel_ned(3) quat(4) rates_body(3) rotor_omega(1) battery_Ah_used(1)
"""
import numpy as np
from .mathutil import q2R, qmul

G = 9.80665


def isa_rho(alt):
    return 1.225 * (1 - 2.25577e-5 * max(alt, 0.0)) ** 4.2559


class Airframe:
    NX = 15

    def __init__(self):
        self.m = 0.85
        self.I = np.diag([0.010, 0.016, 0.024])
        self.Iinv = np.linalg.inv(self.I)
        self.b, self.c = 1.0, 0.20

        # ---------------- lifting strips ----------------
        # columns: r(3) S CLa CL0 CD0 n(3) tau dmax K(3: da,de,dr) cover wash_factor kind
        rows = []

        def add(r, S, CLa, CL0, CD0, n, tau, dmax, K, cover, wf, kind):
            rows.append((r, S, CLa, CL0, CD0, n, tau, dmax, K, cover, wf, kind))

        up = np.array([0.0, 0.0, -1.0])
        cw = lambda y: float(np.clip((0.34 - abs(y)) / 0.16, 0, 1))
        for side, ys in ((-1, -1), (1, 1)):
            # centre panel, inner aileron, outer aileron  (side: -1 left, +1 right)
            add([-0.01, side * 0.06, 0.0], 0.024, 4.25, 0.12, 0.030, up, 0, 0, [0, 0, 0], cw(0.06), 1.7, 0)
            add([-0.01, side * 0.20, 0.0], 0.032, 4.25, 0.12, 0.030, up, 0.5, 0.35, [-side, 0, 0], cw(0.20), 1.7, 0)
            add([-0.01, side * 0.39, 0.0], 0.044, 4.25, 0.12, 0.030, up, 0.5, 0.35, [-side, 0, 0], cw(0.39), 1.7, 0)
        tc = lambda y: float(np.clip((0.26 - abs(y)) / 0.12, 0, 1))
        for side in (-1, 1):
            add([-0.48, side * 0.10, 0.0], 0.034, 3.08, 0.0, 0.020, up, 0.7, 0.50, [0, -1, 0], tc(0.10), 1.4, 1)
        add([-0.47, 0.0, -0.09], 0.090, 2.09, 0.0, 0.020, np.array([0.0, -1.0, 0.0]), 0.7, 0.60, [0, 0, 1], 0.95, 1.4, 2)

        self.N = len(rows)
        self.sr = np.array([r[0] for r in rows], float)
        self.sS = np.array([r[1] for r in rows])
        self.sCLa = np.array([r[2] for r in rows])
        self.sCL0 = np.array([r[3] for r in rows])
        self.sCD0 = np.array([r[4] for r in rows])
        self.sn = np.array([r[5] for r in rows], float)
        self.sc = np.tile(np.array([1.0, 0, 0]), (self.N, 1))
        self.sk = np.array([r[6] * r[7] for r in rows])          # tau * dmax
        self.sK = np.array([r[8] for r in rows], float)
        self.scov = np.array([r[9] for r in rows])
        self.swf = np.array([r[10] for r in rows])
        kind = np.array([r[11] for r in rows])
        self.is_wing, self.is_tail, self.is_fin = kind == 0, kind == 1, kind == 2
        self.sAR = np.where(self.is_wing, 5.0, np.where(self.is_tail, 3.0, 1.5))
        self.sspan = np.cross(self.sc, self.sn)                  # span direction for Cm0
        self.wing_Cm0 = -0.04
        self.a_stall = 0.30

        # fuselage bluff-body drag area per axis (m^2)
        self.fus_CdA = np.array([0.0085, 0.050, 0.065])
        self.fus_r = np.array([0.0, 0.0, 0.0])

        # ---------------- propulsion ----------------
        self.Dp = 0.254
        self.Ap = np.pi * (self.Dp / 2) ** 2
        self.CT0, self.CP0, self.J0 = 0.110, 0.045, 0.55
        self.Jr = 2.0e-5
        self.Kv = 900.0
        self.Ke = 60.0 / (2 * np.pi * self.Kv)             # V/(rad/s) == Nm/A
        self.Rm, self.I0, self.Imax = 0.07, 1.0, 45.0
        self.cells, self.cap_Ah, self.R_int = 4, 2.2, 0.020
        self.I_av = 0.4
        self.I_last = 0.0
        self.Vb_last = 16.0

        # ---------------- landing gear ----------------
        self.pts = [
            (np.array([0.28, 0.0, 0.10]), 'nose'),
            (np.array([-0.04, -0.16, 0.10]), 'main'),
            (np.array([-0.04, 0.16, 0.10]), 'main'),
            (np.array([0.34, 0.0, 0.0]), 'hard'),
            (np.array([0.0, -0.50, -0.02]), 'hard'),
            (np.array([0.0, 0.50, -0.02]), 'hard'),
            (np.array([-0.50, 0.0, 0.0]), 'hard'),
            (np.array([-0.45, 0.0, -0.15]), 'hard'),
        ]
        self.k_gear, self.c_gear = 2000.0, 25.0
        self.crashed = False
        self.crash_reason = ''
        self.on_ground = False

    # ------------------------------------------------------------------ battery / motor
    def v_oc(self, Ah_used):
        s = np.clip(1.0 - Ah_used / self.cap_Ah, 0.0, 1.0)
        return self.cells * (3.30 + 0.90 * np.sqrt(s))

    def soc(self, Ah_used):
        return float(np.clip(1.0 - Ah_used / self.cap_Ah, 0.0, 1.0))

    def prop(self, omega, Vax, rho):
        n = abs(omega) / (2 * np.pi)
        J = Vax / (max(n, 3.0) * self.Dp)
        CT = np.clip(self.CT0 * (1 - J / self.J0), -0.05, self.CT0)
        CP = max(self.CP0 * (1 - 0.5 * J / self.J0), 0.003)
        T = CT * rho * n * n * self.Dp ** 4
        Q = CP / (2 * np.pi) * rho * n * n * self.Dp ** 5
        return T, Q

    # ------------------------------------------------------------------ aerodynamics
    def aero(self, va, w, defl, T, vi, rho, h):
        """Return body force and moment from all strips + fuselage."""
        wash = self.swf * vi * self.scov
        u = va + np.cross(w, self.sr)
        u[:, 0] += wash
        u[self.is_fin, 1] -= 0.08 * wash[self.is_fin]            # slipstream swirl at the fin
        shift = self.sk * (self.sK @ defl)
        # wing downwash on the tail
        al_b = np.arctan2(va[2], max(va[0], 0.5))
        CLw = np.clip(4.25 * al_b + 0.12, -1.2, 1.2)
        shift = shift - 0.15 * CLw * self.is_tail
        ux = u @ np.array([1.0, 0, 0])
        un = np.einsum('ij,ij->i', u, self.sn)
        Vp = np.hypot(ux, un) + 1e-6
        al = np.arctan2(-un, ux)
        ae = al + shift
        sig = 1 / (1 + np.exp(np.clip(-(np.abs(ae) - self.a_stall) / 0.06, -50, 50)))
        ge = np.exp(-max(h, 0.0) / 0.3) * self.is_wing
        CLlin = (self.sCLa * ae + self.sCL0) * (1 + 0.2 * ge)
        CL = (1 - sig) * np.clip(CLlin, -1.5, 1.5) + sig * 2 * np.sin(ae) * np.cos(ae)
        CDi = CLlin ** 2 / (np.pi * 0.8 * self.sAR) * (1 - 0.5 * ge)
        CD = (1 - sig) * (self.sCD0 + CDi) + sig * (self.sCD0 + 2.0 * np.sin(ae) ** 2)
        q = 0.5 * rho * Vp * Vp
        wh = (ux[:, None] * self.sc + un[:, None] * self.sn) / Vp[:, None]
        ld = self.sn - np.einsum('ij,ij->i', self.sn, wh)[:, None] * wh
        nl = np.linalg.norm(ld, axis=1)
        ld = ld / np.maximum(nl, 1e-6)[:, None]
        F = (q * self.sS)[:, None] * (CL[:, None] * ld - CD[:, None] * wh)
        M = np.cross(self.sr, F)
        M += (q * self.sS * self.c * self.wing_Cm0 * self.is_wing)[:, None] * self.sspan
        Ft, Mt = F.sum(0), M.sum(0)
        # fuselage: bluff body drag along each axis
        Ff = -0.5 * rho * np.abs(va) * va * self.fus_CdA
        return Ft + Ff, Mt + np.cross(self.fus_r, Ff), (ae, CL)

    # ------------------------------------------------------------------ ground contact
    def contact(self, pos, vel, R, w, steer, brake):
        F = np.zeros(3)
        M = np.zeros(3)
        touching = False
        for r, kind in self.pts:
            pw = pos + R @ r
            pen = pw[2]
            if pen <= 0:
                continue
            vw = vel + R @ np.cross(w, r)
            N = max(0.0, self.k_gear * pen + self.c_gear * vw[2])
            touching = True
            if kind == 'hard' and np.linalg.norm(vw) > 4.0 and not self.crashed:
                self.crashed = True
                self.crash_reason = f'hard point {r} hit ground at {np.linalg.norm(vw):.1f} m/s (vz {vw[2]:.1f})'
            if kind != 'hard' and vw[2] > 5.0 and not self.crashed:
                self.crashed = True
                self.crash_reason = f'gear {kind} impact vz {vw[2]:.1f} m/s'
            if kind == 'hard':
                vt = vw[:2]
                Ft = -0.6 * N * vt / (np.linalg.norm(vt) + 0.1)
            else:
                f = R[:2, 0].copy()
                ang = steer if kind == 'nose' else 0.0
                ca, sa = np.cos(ang), np.sin(ang)
                f = np.array([ca * f[0] - sa * f[1], sa * f[0] + ca * f[1]])
                nf = np.linalg.norm(f)
                f = f / nf if nf > 1e-6 else np.array([1.0, 0.0])
                lat = np.array([-f[1], f[0]])
                vf, vl = vw[:2] @ f, vw[:2] @ lat
                mu_roll = 0.03 + (0.35 if kind == 'main' else 0.0) * brake
                Ft = -(0.9 * N * np.tanh(vl / 0.15)) * lat - (mu_roll * N * np.tanh(vf / 0.15)) * f
            Fw = np.array([Ft[0], Ft[1], -N])
            Fb = R.T @ Fw
            F += Fb
            M += np.cross(r, Fb)
        self.on_ground = touching
        return F, M

    # ------------------------------------------------------------------ dynamics
    def deriv(self, x, u, wind, rho=1.225):
        pos, vel, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        om, Ah = x[13], x[14]
        da, de, dr, duty, brake = u
        R = q2R(q)
        va = R.T @ (vel - wind)
        Vax = max(va[0], 0.0)

        # motor / prop / battery
        Vb = self.v_oc(Ah) - self.R_int * self.I_last
        Vm = np.clip(duty, 0, 1) * Vb
        Im = np.clip((Vm - om * self.Ke) / self.Rm, 0.0, self.Imax)
        Qm = self.Ke * max(Im - self.I0, 0.0) if Im > 0 else 0.0
        T, Qp = self.prop(om, Vax, rho)
        om_dot = (Qm - Qp - 2.0e-6 * om) / self.Jr
        vi = -Vax / 2 + np.sqrt((Vax / 2) ** 2 + max(T, 0.0) / (2 * rho * self.Ap))
        self.I_last = Im + self.I_av
        self.Vb_last = Vb

        Fa, Ma, aux = self.aero(va, w, np.array([da, de, dr]), T, vi, rho, -pos[2] + 0.05)
        M_prop = np.array([-Qm, 0.0, 0.0]) - np.cross(w, np.array([self.Jr * om, 0.0, 0.0]))

        if pos[2] > -1.0:
            Fc, Mc = self.contact(pos, vel, R, w, 0.45 * dr, brake)
        else:
            Fc, Mc = np.zeros(3), np.zeros(3)
            self.on_ground = False

        F_ng = Fa + np.array([T, 0.0, 0.0]) + Fc
        M = Ma + M_prop + Mc
        xd = np.empty(self.NX)
        xd[0:3] = vel
        xd[3:6] = R @ (F_ng / self.m) + np.array([0, 0, G])
        xd[6:10] = 0.5 * qmul(q, np.array([0.0, *w]))
        xd[10:13] = self.Iinv @ (M - np.cross(w, self.I @ w))
        xd[13] = om_dot
        xd[14] = (Im + self.I_av) / 3600.0
        self.info = dict(T=T, I=Im, Vb=Vb, rpm=om * 60 / (2 * np.pi), Q=Qp, vi=vi, ae=aux[0])
        return xd, F_ng / self.m

    def step(self, x, u, wind, dt, rho=1.225):
        k1, f = self.deriv(x, u, wind, rho)
        k2, _ = self.deriv(x + 0.5 * dt * k1, u, wind, rho)
        k3, _ = self.deriv(x + 0.5 * dt * k2, u, wind, rho)
        k4, _ = self.deriv(x + dt * k3, u, wind, rho)
        xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[6:10] /= np.linalg.norm(xn[6:10])
        xn[13] = max(xn[13], 0.0)
        return xn, f, self.info['T']
