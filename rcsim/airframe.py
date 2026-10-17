"""Physics v3: 6-DOF rigid body in a fluid, built up from a component mass breakdown.

Frames: body FRD (x nose, y right, z down) with the origin at the centre of gravity, world NED,
ground at z = 0.  All geometry below is written in a fixed build datum and shifted to the CG that
the component list produces, so moving the battery really moves the CG.

What is modelled and why
------------------------
Mass / inertia     m, CG and the full inertia tensor (incl. Ixz) are summed from the parts list
                   (motor, battery, wing foam, servos, ...), not guessed.
Rigid body         Newton-Euler in body axes with the spinning rotor's angular momentum h:
                       (M_RB + M_A) nu' = tau - C_RB(nu) nu
                   C_RB = [m w x v ; w x (I w + h)]  (centripetal + gyroscopic terms)
Added mass         A flat plate accelerating through air drags a slab of air with it
                   (rho*pi*c^2/4 per unit span).  For a foamie this adds ~25 % to roll inertia.
                   Summed per strip into a 6x6 matrix M_A acting on the air-relative acceleration.
                   Only the acceleration part is used: the steady (Munk) part of the potential-flow
                   loads is already inside the measured plate coefficients, adding it again makes a
                   stalled plate fall flat.  C_A does no work, so energy is still conserved.
Aerodynamics       Strip theory: each wing / tail / fin panel sees its own local flow
                   (free stream + w x r + prop slipstream + slipstream swirl), so damping, adverse
                   yaw and roll/yaw coupling come out of the geometry.  Finite-span lift slope
                   (Helmbold), induced drag, gentle Viterna-Corrigan post-stall to a finite-plate
                   normal force (CN90 ~ 1.2, not the 2-D 2.0), centre of pressure walks from c/4 to c/2 in
                   stall, ground effect per strip height (McCormick), downwash at the tail lags
                   the wing by the convection time (alpha-dot damping).
Fuselage           Two bluff segments fore and aft, so it damps yaw/pitch rotation, and the
                   front one sits in the slipstream (scrubbing drag).
Slipstream         Momentum-theory jet (vi at the disc -> 2 vi, contracting), then spreading and slowing
                   by turbulent mixing at constant momentum flux.  Each strip is split into the part
                   inside the jet and the part outside.  Swirl from the prop torque (Q = mdot w R^2/2).
Propulsion         LiPo (OCV(SoC) + internal resistance) -> ESC (Ib = duty * Im, power conserved)
                   -> brushless motor (Kv, Rm, I0 friction torque always acting) -> prop CT/CP(J)
                   -> rotor inertia.  Battery sag is solved algebraically, no one-step lag.
                   Prop in-plane (H) force in crossflow; a freewheeling prop windmills.
Ground             Spring-damper legs + tyre friction with a stick/slip bristle model: brakes and
                   side-grip hold the aeroplane still without teleporting it.
Energy             energy(x) returns kinetic (translational, rotational, rotor, added-air), potential
                   and electrical terms; info['P_ext'] is the power of all external forces, so
                   d(E_mech)/dt = P_ext can be checked (see tests/physics_check.py).

Control inputs (normalised surface positions, -1..1): da>0 right wing down, de>0 nose up, dr>0 nose right.
State x: pos(3) vel_ned(3) quat(4) rates_body(3) rotor_omega_rel(1) battery_Ah_used(1) tail_downwash(1)
"""
import numpy as np
from .mathutil import q2R, qmul

G = 9.80665
GRAV = np.array([0.0, 0.0, G])
EX = np.array([1.0, 0.0, 0.0])


def _cross(a, b):
    """Cross product without np.cross's generality overhead (it was half the simulation time)."""
    if np.ndim(a) == 1 and np.ndim(b) == 1:
        ax, ay, az = a
        bx, by, bz = b
        return np.array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
    a = np.asarray(a)
    b = np.asarray(b)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def isa_rho(alt):
    return 1.225 * (1 - 2.25577e-5 * max(alt, 0.0)) ** 4.2559


def helmbold(AR):
    """Finite-wing lift-curve slope per rad."""
    return 2 * np.pi * AR / (2 + np.sqrt(AR * AR + 4))


def box_inertia(m, size):
    lx, ly, lz = size
    return m / 12.0 * np.diag([ly * ly + lz * lz, lx * lx + lz * lz, lx * lx + ly * ly])


# ---------------------------------------------------------------------------------------------
# Parts list, build datum (FRD).  (name, kg, position m, box size m)  -- box = uniform-density block
# ---------------------------------------------------------------------------------------------
COMPONENTS = [
    ('motor 4S 900Kv outrunner',   0.105, (0.290, 0.000, 0.000), (0.040, 0.035, 0.035)),
    ('esc 40A',                    0.040, (0.100, 0.000, 0.022), (0.060, 0.025, 0.010)),
    ('battery 4S 2200',            0.235, (-0.070, 0.000, 0.028), (0.105, 0.034, 0.035)),
    ('rx + fc + gps',              0.045, (-0.100, 0.000, -0.025), (0.050, 0.030, 0.020)),
    ('aileron servo L',            0.009, (-0.040, -0.220, 0.000), (0.023, 0.012, 0.022)),
    ('aileron servo R',            0.009, (-0.040, 0.220, 0.000), (0.023, 0.012, 0.022)),
    ('elevator servo',             0.009, (-0.120, 0.000, 0.020), (0.023, 0.012, 0.022)),
    ('rudder servo',               0.009, (-0.120, 0.000, -0.020), (0.023, 0.012, 0.022)),
    ('wing foam + spars',          0.130, (-0.060, 0.000, 0.000), (0.200, 1.000, 0.006)),
    ('profile fuselage',           0.068, (-0.080, 0.000, 0.000), (0.900, 0.006, 0.150)),
    ('horizontal tail',            0.022, (-0.500, 0.000, 0.000), (0.170, 0.400, 0.004)),
    ('fin + rudder',               0.012, (-0.500, 0.000, -0.090), (0.250, 0.004, 0.200)),
    ('main gear + wheels',         0.042, (-0.040, 0.000, 0.070), (0.030, 0.320, 0.070)),
    ('nose gear',                  0.015, (0.280, 0.000, 0.070), (0.020, 0.030, 0.070)),
    ('linkages, wiring, glue',     0.050, (-0.040, 0.000, 0.000), (0.500, 0.200, 0.040)),
]
PROP_POS = (0.320, 0.000, 0.000)
PROP_MASS = 0.020

FIN_S = 0.050   # fin + rudder area m^2 = 25 % of wing area (3D foamies 15-25 %)


class Airframe:
    NX = 16

    def __init__(self):
        # ---------------- mass ----------------
        self.components = [(n, m, np.array(p, float), np.array(s, float)) for n, m, p, s in COMPONENTS]
        self.prop_pos0 = np.array(PROP_POS, float)
        self.prop_mass = PROP_MASS

        # ---------------- geometry (datum) ----------------
        self.b, self.c = 1.0, 0.20
        self.AR_w = self.b ** 2 / (self.b * self.c)
        S_t, b_t = 0.068, 0.40
        AR_t = b_t ** 2 / S_t
        AR_f = 2 * 0.2 ** 2 / FIN_S                # fin on a fuselage/stab: end-plate doubles its AR

        # lifting strips: r (aero centre, c/4) | span | chord | normal | CLa | AR(induced) | CD0 |
        #                 a_stall | CN90 | flap (tau, max throw rad) | K (da,de,dr) | slipstream profile | kind
        rows = []
        up = (0.0, 0.0, -1.0)

        def add(r, span, chord, n, AR, CD0, a_stall, CN90, flap, K, mix, kind):
            rows.append(dict(r=r, span=span, chord=chord, n=n, CLa=helmbold(AR), AR=AR, CD0=CD0,
                             a_stall=a_stall, CN90=CN90, tau=flap[0], dmax=flap[1], K=K, mix=mix, kind=kind))

        for side in (-1, 1):
            # centre panel (fuselage side), inner and outer aileron: 3D-style ailerons start 5 cm from
            # the fuselage so their roots sit in the prop wash (the only roll control in a hover)
            add((-0.01, side * 0.025, 0.0), 0.05, self.c, up, self.AR_w, 0.020, 0.28, 1.20, (0.0, 0.0), (0, 0, 0), 0.9, 0)
            add((-0.01, side * 0.150, 0.0), 0.20, self.c, up, self.AR_w, 0.020, 0.28, 1.20, (0.5, 0.70), (-side, 0, 0), 0.9, 0)
            add((-0.01, side * 0.375, 0.0), 0.25, self.c, up, self.AR_w, 0.020, 0.28, 1.20, (0.5, 0.70), (-side, 0, 0), 0.9, 0)
        for side in (-1, 1):
            add((-0.48, side * 0.10, 0.0), 0.20, 0.17, up, AR_t, 0.018, 0.32, 1.17, (0.7, 0.50), (0, -1, 0), 0.9, 1)
        add((-0.47, 0.0, -0.09), 0.20, FIN_S / 0.20, (0.0, -1.0, 0.0), AR_f, 0.018, 0.38, 1.17, (0.7, 0.60), (0, 0, 1), 0.9, 2)
        self.strip_rows = rows

        # fuselage: bluff segments (datum position, CdA along x,y,z m^2)
        self.fus_r0 = np.array([[0.20, 0.0, 0.0], [-0.30, 0.0, 0.0]])
        self.fus_CdA = np.array([[0.0030, 0.025, 0.010], [0.0020, 0.035, 0.010]])   # front incl. gear, motor

        # ---------------- propulsion ----------------
        self.Dp = 0.254
        self.Ap = np.pi * (self.Dp / 2) ** 2
        self.CT0, self.CP0, self.J0 = 0.110, 0.045, 0.55
        self.kH = 0.6                                      # prop in-plane force / (mdot * crossflow)
        self.Jr = 6.4e-5                                   # prop (5.5e-5) + outrunner bell (0.9e-5)
        self.Kv = 900.0
        self.Rm, self.I0, self.Imax = 0.07, 1.0, 45.0
        self.B_visc = 2.0e-7                               # bearing/windage viscous torque Nm/(rad/s)
        self.cells, self.cap_Ah, self.R_int = 4, 2.2, 0.020
        self.I_av = 0.4                                    # receiver, servos, FC

        # ---------------- landing gear / ground ----------------
        self.pts0 = [
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
        self.k_tan, self.c_tan = 1500.0, 12.0              # tyre / skid tangential stiffness, damping
        self.mu_lat, self.mu_roll, self.mu_brake, self.mu_slide = 0.70, 0.04, 0.55, 0.50
        self.crashed = False
        self.crash_reason = ''
        self.on_ground = False
        self.morph = dict(ail=1.0, ele=1.0, rud=1.0, wing=1.0, drag=1.0)

        self._build()
        self.info = dict(T=0.0, I=0.0, Vb=self.v_oc(0.0), rpm=0.0, Q=0.0, vi=0.0, ae=np.zeros(self.N), P_ext=0.0)

    # ------------------------------------------------------------------ derived model
    def _build(self):
        """Mass properties from the parts list, then everything re-referenced to the CG."""
        m = sum(c[1] for c in self.components) + self.prop_mass
        cg = (sum(c[1] * c[2] for c in self.components) + self.prop_mass * self.prop_pos0) / m
        I = np.zeros((3, 3))
        for _, mi, p, size in self.components:
            d = p - cg
            I += box_inertia(mi, size) + mi * (d @ d * np.eye(3) - np.outer(d, d))
        d = self.prop_pos0 - cg
        I += np.diag([0.0, self.Jr / 2, self.Jr / 2]) + self.prop_mass * (d @ d * np.eye(3) - np.outer(d, d))
        # the rotor's own spin inertia (Jr about x) is carried by the rotor state, not by I
        self.m, self.cg_datum, self.I = m, cg, I
        self.Iinv = np.linalg.inv(I)
        self.rp = self.prop_pos0 - cg
        self.Ke = 60.0 / (2 * np.pi * self.Kv)             # V/(rad/s) == Nm/A
        self.M_RB = np.zeros((6, 6))
        self.M_RB[:3, :3] = m * np.eye(3)
        self.M_RB[3:, 3:] = I
        self._build_aero()
        self.fus_r = self.fus_r0 - cg
        self.pts = [(p - cg, k) for p, k in self.pts0]
        self.zc = np.zeros((len(self.pts), 2))              # tangential bristle deflection, world xy

    def _build_aero(self):
        """Strip geometry and coefficients from the build rows with the in-flight morph applied
        (no change to mass, CG or the flight state, so it can be called while flying)."""
        cg = self.cg_datum
        mo = self.morph
        rows = self.strip_rows
        self.N = len(rows)
        chord, tau, cd0 = [], [], []
        for r in rows:
            k = r['K']
            ks = mo['ail'] if k[0] else (mo['ele'] if k[1] else (mo['rud'] if k[2] else None))
            chord.append(r['chord'] * (mo['wing'] if r['kind'] == 0 else 1.0))
            # a bigger control surface (larger flap chord): more effectiveness (thin-aerofoil flap theory
            # goes roughly with sqrt of the flap-chord ratio), a little more profile drag (gaps, area)
            tau.append(min(r['tau'] * ks ** 0.6, 0.95) if ks else r['tau'])
            cd0.append(r['CD0'] + (0.004 * (ks - 1.0) if ks else 0.0))
        self.CdA_plate = 0.006 * max(mo['drag'] - 1.0, 0.0)   # airbrake plate on top of the fuselage
        self.sr = np.array([r['r'] for r in rows], float) - cg
        self.sspan = np.array([r['span'] for r in rows])
        self.schord = np.array(chord)
        self.sS = self.sspan * self.schord
        self.sn = np.array([r['n'] for r in rows], float)
        self.sc = np.tile(EX, (self.N, 1))
        self.sCLa = np.array([r['CLa'] for r in rows])
        self.sAR = np.array([r['AR'] for r in rows]) / np.where(np.array([r['kind'] for r in rows]) == 0, mo['wing'], 1.0)
        self.sCD0 = np.array(cd0)
        self.sastall = np.array([r['a_stall'] for r in rows])
        self.sCN90 = np.array([r['CN90'] for r in rows])
        self.stau = np.array(tau)
        self.sdmax = np.array([r['dmax'] for r in rows])
        self.sK = np.array([r['K'] for r in rows], float)
        self.smix = np.array([r['mix'] for r in rows])
        kind = np.array([r['kind'] for r in rows])
        self.is_wing, self.is_tail, self.is_fin = kind == 0, kind == 1, kind == 2
        t2 = lambda a: np.concatenate([a, a, a])
        self.t_n, self.t_c, self.t_r, self.t_chord = t2(self.sn), t2(self.sc), t2(self.sr), t2(self.schord)
        self.t_CLa, self.t_AR, self.t_CD0 = t2(self.sCLa), t2(self.sAR), t2(self.sCD0)
        self.t_astall, self.t_CN90 = t2(self.sastall), t2(self.sCN90)
        a_s = self.sastall
        CLs = self.sCLa * a_s
        CDs = self.sCD0 + CLs ** 2 / (np.pi * 0.8 * self.sAR)
        self.t_VA = t2((CLs - self.sCN90 * np.sin(a_s) * np.cos(a_s)) * np.sin(a_s) / np.cos(a_s) ** 2)
        self.t_VB = t2((CDs - self.sCN90 * np.sin(a_s) ** 2) / np.cos(a_s))
        self.l_tail = float(np.mean(self.sr[self.is_wing, 0]) - np.mean(self.sr[self.is_tail, 0]))

        # slipstream geometry: distance aft of the disc, perpendicular offset of each strip from the
        # prop axis, and the near edge of the strip measured from the axis along its span
        rel = self.sr - self.rp
        self.s_daft = np.maximum(-rel[:, 0], 0.0)
        self.s_perp = rel.copy()
        self.s_perp[:, 0] = 0.0
        sdir = _cross(self.sc, self.sn)
        centre_off = np.abs(np.einsum('ij,ij->i', self.s_perp, sdir))
        self.s_d0 = np.maximum(centre_off - self.sspan / 2, 0.0)
        self.s_dev = 1.0 + self.s_daft / np.sqrt(self.s_daft ** 2 + (self.Dp / 2) ** 2)

        # added (apparent) mass: each plate carries rho*pi*c^2/4*span*k_AR of air along its normal,
        # acting at mid-chord, plus the plate's own pitching added inertia rho*pi*c^4/128*span
        self.A1 = np.zeros((6, 6))                          # per unit air density
        r_mid = self.sr - 0.25 * self.schord[:, None] * self.sc
        for i in range(self.N):
            AR_geo = self.sAR[i] if not self.is_fin[i] else 0.8
            k = AR_geo / np.sqrt(1 + AR_geo ** 2)
            a = np.pi / 4 * self.schord[i] ** 2 * self.sspan[i] * k
            g = np.concatenate([self.sn[i], _cross(r_mid[i], self.sn[i])])
            self.A1 += a * np.outer(g, g)
            sdir_i = _cross(self.sc[i], self.sn[i])
            ar = np.pi / 128 * self.schord[i] ** 4 * self.sspan[i]
            self.A1[3:, 3:] += ar * np.outer(sdir_i, sdir_i)
        self._rho_cached = None

    MORPH_KEYS = ('ail', 'ele', 'rud', 'wing', 'drag')

    def set_morph(self, **kw):
        """Change the aeroplane in flight: ail/ele/rud = control-surface size (chord) factor, wing = wing
        chord (lift, area) factor, drag = 1 + extra drag from an airbrake plate.  Mass and CG are unchanged."""
        for k, v in kw.items():
            if k not in self.MORPH_KEYS:
                raise KeyError(k)
            self.morph[k] = float(v)
        self._build_aero()

    def _Minv(self, rho):
        if rho != self._rho_cached:
            self._rho_cached = rho
            self.A = rho * self.A1
            self.Minv6 = np.linalg.inv(self.M_RB + self.A)
        return self.Minv6

    def randomize(self, rng, k=1.0):
        """Build-to-build variation (k = 0 off, 1 typical hobby build): part weights, battery slot (CG),
        aero and propulsion tolerances.  The autopilot is tuned on the nominal model and has to cope."""
        g = lambda sd: 1.0 + k * sd * rng.normal()
        comps = []
        for n, m, p, s in self.components:
            p = p.copy()
            if n.startswith('battery'):
                p[0] += k * 0.010 * rng.normal()            # where the pilot straps the pack in
            comps.append((n, m * g(0.04), p, s))
        self.components = comps
        fCLa, fCD0, fK, fst = g(0.03), g(0.08), g(0.05), g(0.04)
        for r in self.strip_rows:
            r['CLa'] *= fCLa
            r['CD0'] *= fCD0
            r['tau'] *= fK
            r['a_stall'] *= fst
        self.fus_CdA = self.fus_CdA * g(0.10)
        self.CT0 *= g(0.04)
        self.CP0 *= g(0.04)
        self.Kv *= g(0.02)
        self.R_int *= g(0.15)
        self.cap_Ah *= g(0.05)
        self.k_gear *= g(0.1)
        self._build()

    def modify(self, ail=1.0, ele=1.0, rud=1.0, mass=1.0, thrust=1.0, scale=1.0, surf=1.0, wing=1.0):
        """A different aeroplane, not told to the flight controller:
        ail/ele/rud  control power of each axis      surf   size of all control surfaces
        mass         every part's mass (weight and inertia)
        thrust       prop thrust coefficient          wing   wing chord (area) factor
        scale        whole aeroplane bigger/smaller: every length x scale (prop too), mass x scale^2.5
                     (how RC models of one type scale), gear stiffness with it."""
        k = scale
        km = mass * k ** 2.5
        for r in self.strip_rows:
            kk = r['K']
            r['tau'] *= surf * (ail if kk[0] else (ele if kk[1] else (rud if kk[2] else 1.0)))
            r['r'] = tuple(np.array(r['r']) * k)
            r['span'] *= k
            r['chord'] *= k * (wing if r['kind'] == 0 else 1.0)
        self.components = [(n, m * km, p * k, s * k) for n, m, p, s in self.components]
        self.prop_pos0 = self.prop_pos0 * k
        self.prop_mass *= km
        self.Dp *= k
        self.Ap = np.pi * (self.Dp / 2) ** 2
        self.Jr *= km * k * k
        self.b *= k
        self.c *= k * wing
        self.AR_w = self.b / self.c
        self.fus_r0 = self.fus_r0 * k
        self.fus_CdA = self.fus_CdA * k * k
        self.pts0 = [(p * k, kind) for p, kind in self.pts0]
        self.k_gear *= km
        self.c_gear *= km
        self.k_tan *= km
        self.c_tan *= km
        # a bigger model gets a proportionally bigger motor and pack (same cell count)
        self.Imax *= km
        self.I0 *= km
        self.Rm /= km
        self.R_int /= km
        self.cap_Ah *= km
        self.CT0 *= thrust                     # thrust per watt (prop choice); thrust^1.5 in hover terms
        self._build()

    def initial_state(self, pos=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0)):
        x = np.zeros(self.NX)
        x[0:3] = pos
        x[6:10] = q
        return x

    # ------------------------------------------------------------------ battery / motor / prop
    def v_oc(self, Ah_used):
        s = np.clip(1.0 - Ah_used / self.cap_Ah, 0.0, 1.0)
        return self.cells * (3.30 + 0.90 * np.sqrt(s))

    def soc(self, Ah_used):
        return float(np.clip(1.0 - Ah_used / self.cap_Ah, 0.0, 1.0))

    def prop(self, omega, Vax, rho):
        """Thrust and absorbed torque.  CT falls linearly with advance ratio (negative = windmill drag),
        CP has a profile-drag floor and turns negative (prop drives the motor) when windmilling."""
        n = max(omega, 0.0) / (2 * np.pi)
        # a slow or stopped prop in an airstream: forces scale with V^2 instead of n^2 (J capped at 2 J0),
        # so a freewheeling prop is spun up by the air and windmills (drag + driving torque)
        n_eff = max(n, Vax / (2 * self.J0 * self.Dp), 1e-3)
        J = Vax / (n_eff * self.Dp)
        CT = max(self.CT0 * (1 - J / self.J0), -0.08)
        jj = J / self.J0
        CP = max(self.CP0 * (1 + 0.3 * jj - 0.9 * jj * jj), -0.05)
        T = CT * rho * n_eff * n_eff * self.Dp ** 4
        Q = CP / (2 * np.pi) * rho * n_eff * n_eff * self.Dp ** 5
        return T, Q

    def electrics(self, duty, Om, Ah):
        """ESC (averaging buck converter, freewheel diode: no regen) + motor + battery sag.
        Vb = Voc - R_int*(d*Im + I_av),  Im = (d*Vb - Ke*Om)/Rm  ->  solved together."""
        d = float(np.clip(duty, 0.0, 1.0))
        Voc = self.v_oc(Ah)
        Im = (d * (Voc - self.R_int * self.I_av) - self.Ke * Om) / (self.Rm + d * d * self.R_int)
        Im = float(np.clip(Im, 0.0, self.Imax))
        Ib = d * Im + self.I_av
        return Im, Ib, Voc - self.R_int * Ib

    # ------------------------------------------------------------------ aerodynamics
    def aero(self, va, w, defl, vi, Vax, Qp, Om, rho, h_strip, eps):
        """Strip + fuselage forces/moments.  Returns force, moment, power (inertial frame, still air
        -> with wind the power is relative to the air), tail-downwash target and its time constant."""
        # Slipstream.  Momentum theory: the jet speeds up from vi at the disc to 2 vi and contracts; then
        # turbulent mixing spreads it (radius grows ~0.1 per unit length) and slows it, conserving momentum
        # flux.  Every strip is split into the part inside the jet and the part in free stream.
        Rs = (self.Dp / 2) * np.sqrt((Vax + vi + 1e-6) / (Vax + vi * self.s_dev + 1e-6))
        Rj = Rs + 0.1 * self.s_daft
        decay = Rs / Rj
        dv = self.smix * vi * self.s_dev * decay
        cover = np.clip((Rj - self.s_d0) / self.sspan, 0.0, 1.0)
        # swirl: angular momentum flux of the jet balances the prop torque, spread out the same way
        mdot = rho * self.Ap * (Vax + vi)
        om_s = 2 * Qp / max(mdot * (self.Dp / 2) ** 2, 1e-4)        # Q = mdot * om * R^2 / 2
        om_s = float(np.clip(om_s, -0.3 * Om, 0.3 * Om))
        rad = np.linalg.norm(self.s_perp, axis=1)
        sc = np.minimum(1.0, 0.7 * Rj / np.maximum(rad, 1e-6))
        v_swirl = (om_s * ((self.Dp / 2) / Rj) ** 2)[:, None] * _cross(EX, self.s_perp * sc[:, None])

        u0 = va + _cross(w, self.sr)
        u1 = u0.copy()
        u1[:, 0] += dv
        u = np.vstack([u0, u1, u1 - v_swirl])
        S = np.concatenate([self.sS * (1 - cover), self.sS * cover, self.sS * cover])
        n2, c2 = self.t_n, self.t_c
        # control surfaces: plain-flap effectiveness tau, falling off at large throws (~-35 % at 40 deg)
        d = self.sdmax * (self.sK @ defl)
        shift = np.tile(self.stau * d * (1 - 0.35 * (d / 0.70) ** 2) - eps * self.is_tail, 3)
        ux = u[:, 0]
        un = np.einsum('ij,ij->i', u, n2)
        Vp = np.hypot(ux, un) + 1e-6
        al = np.arctan2(-un, ux)
        ae = al + shift
        sig = 1 / (1 + np.exp(np.clip(-(np.abs(ae) - self.t_astall) / 0.05, -50, 50)))
        hb = 16 * np.maximum(h_strip, 0.02) / self.b
        phi = np.tile(np.where(self.is_wing, hb * hb / (1 + hb * hb), 1.0), 3)   # McCormick ground effect
        CLlin = np.clip(self.t_CLa * ae * (1 + 0.1 * (1 - phi)), -1.4, 1.4)
        CDi = CLlin ** 2 / (np.pi * 0.8 * self.t_AR) * phi
        # post stall: Viterna-Corrigan flat-plate curves, matched to the attached curve at the stall angle
        ap = np.where(ae >= 0, 1.0, -1.0) * np.maximum(np.abs(ae), self.t_astall)
        sa, ca = np.sin(ap), np.cos(ap)
        vit = self.t_VA * ca * ca / np.where(np.abs(sa) > 1e-3, sa, 1e-3)
        CLp = self.t_CN90 * sa * ca + np.where(np.abs(ap) < np.pi / 2, vit, 0.0)
        CDp = self.t_CN90 * sa * sa + self.t_VB * np.abs(ca)
        CL = (1 - sig) * CLlin + sig * CLp
        CD = (1 - sig) * (self.t_CD0 + CDi) + sig * CDp
        q = 0.5 * rho * Vp * Vp
        wh = (ux[:, None] * c2 + un[:, None] * n2) / Vp[:, None]
        ld = n2 - np.einsum('ij,ij->i', n2, wh)[:, None] * wh
        ld = ld / np.maximum(np.linalg.norm(ld, axis=1), 1e-6)[:, None]
        F = (q * S)[:, None] * (CL[:, None] * ld - CD[:, None] * wh)
        r_ap = self.t_r - (0.25 * sig * self.t_chord)[:, None] * c2   # c.p. walks to mid-chord in stall
        M = _cross(r_ap, F)
        # The surfaces can only take out angular momentum the prop put into the jet (strip theory, which
        # ignores that the wing straightens the swirl, would let them out-torque the motor): cap the
        # recovered swirl moment at 60 % of the prop torque.
        N = self.N
        dM = M[2 * N:] - M[N:2 * N]
        dMx = float(dM[:, 0].sum())
        k = min(1.0, 0.6 * abs(Qp) / max(abs(dMx), 1e-9))
        F = np.vstack([F[:N], F[N:2 * N] + k * (F[2 * N:] - F[N:2 * N])])
        M = np.vstack([M[:N], M[N:2 * N] + k * dM])
        q, S, CL = q[:2 * N], S[:2 * N], CL[:2 * N]
        v_ap = va + _cross(w, r_ap[:2 * N])
        P = float(np.einsum('ij,ij->', F, v_ap))
        Ft, Mt = F.sum(0), M.sum(0)

        # fuselage segments (front one in the slipstream)
        for j in range(2):
            uf = va + _cross(w, self.fus_r[j])
            vf = uf.copy()
            if j == 0:
                vf[0] += vi * 1.5
            Ff = -0.5 * rho * np.abs(vf) * vf * self.fus_CdA[j]
            Ft += Ff
            Mt += _cross(self.fus_r[j], Ff)
            P += float(Ff @ uf)

        if self.CdA_plate > 0.0:                          # airbrake plate on top of the fuselage
            rpl = np.array([-0.02, 0.0, -0.045 - 0.5 * np.sqrt(self.CdA_plate / 1.2)])
            up = va + _cross(w, rpl)
            Fp = -0.5 * rho * self.CdA_plate * np.linalg.norm(up) * up
            Ft += Fp
            Mt += _cross(rpl, Fp)
            P += float(Fp @ up)

        # tail downwash target from the wing's mean lift coefficient, lagged by convection time
        wing2 = np.tile(self.is_wing, 2)
        qs = q[wing2] * S[wing2]
        CLw = float(np.sum(CL[wing2] * qs) / max(np.sum(qs), 1e-6))
        eps_ss = float(np.clip(1.6 * CLw / (np.pi * self.AR_w), -0.25, 0.25))
        Vw = np.sqrt(2 * max(np.sum(qs), 0.0) / (rho * np.sum(self.sS[self.is_wing])))
        tau = self.l_tail / max(Vw, 2.0)
        return Ft, Mt, P, eps_ss, tau, np.where(cover > 0.5, ae[2 * N:], ae[:N])

    # ------------------------------------------------------------------ ground contact
    def contact(self, pos, vel, R, w, steer, brake, zc=None):
        """Normal: spring-damper per point.  Tangential: bristle spring (stick) saturating at mu*N (slip).
        Wheels roll freely along their heading (rolling resistance), grip sideways, brake on the mains."""
        zc = self.zc if zc is None else zc
        F = np.zeros(3)
        M = np.zeros(3)
        P = 0.0
        for i, (r, kind) in enumerate(self.pts):
            pw = pos + R @ r
            pen = pw[2]
            if pen <= 0:
                continue
            vw = vel + R @ _cross(w, r)
            N = max(0.0, self.k_gear * pen + self.c_gear * vw[2])
            if N <= 0.0:
                continue
            v2, z = vw[:2], zc[i]
            if kind == 'hard':
                Ft = -self.k_tan * z - self.c_tan * v2
                lim = self.mu_slide * N
                nf = np.linalg.norm(Ft)
                if nf > lim:
                    Ft *= lim / nf
            else:
                f = self._wheel_dir(R, steer if kind == 'nose' else 0.0)
                lat = np.array([-f[1], f[0]])
                vf, vl = v2 @ f, v2 @ lat
                zf, zl = z @ f, z @ lat
                Fl = float(np.clip(-self.k_tan * zl - self.c_tan * vl, -self.mu_lat * N, self.mu_lat * N))
                Ff = -self.mu_roll * N * np.tanh(vf / 0.05)
                if kind == 'main' and brake > 0.0:
                    lim = self.mu_brake * brake * N
                    Ff += float(np.clip(-self.k_tan * zf - self.c_tan * vf, -lim, lim))
                Ft = Fl * lat + Ff * f
            Fw = np.array([Ft[0], Ft[1], -N])
            Fb = R.T @ Fw
            F += Fb
            M += _cross(r, Fb)
            P += float(Fw @ vw)
        return F, M, P

    @staticmethod
    def _wheel_dir(R, ang):
        f = R[:2, 0].copy()
        ca, sa = np.cos(ang), np.sin(ang)
        f = np.array([ca * f[0] - sa * f[1], sa * f[0] + ca * f[1]])
        nf = np.linalg.norm(f)
        return f / nf if nf > 1e-6 else np.array([1.0, 0.0])

    def _update_contacts(self, x, dt, steer, brake):
        """After each step: advance the bristles (stick), saturate them (slip), flag crashes."""
        pos, vel, R, w = x[0:3], x[3:6], q2R(x[6:10]), x[10:13]
        touching = False
        if pos[2] < -1.0:
            self.zc[:] = 0.0
            self.on_ground = False
            return
        for i, (r, kind) in enumerate(self.pts):
            pw = pos + R @ r
            if pw[2] <= 0:
                self.zc[i] = 0.0
                continue
            vw = vel + R @ _cross(w, r)
            N = max(0.0, self.k_gear * pw[2] + self.c_gear * vw[2])
            touching = True
            sp = float(np.linalg.norm(vw))
            if kind == 'hard' and sp > 4.0 and not self.crashed:
                self.crashed = True
                self.crash_reason = f'hard point {np.round(r, 2)} hit ground at {sp:.1f} m/s (vz {vw[2]:.1f})'
            if kind != 'hard' and vw[2] > 5.0 and not self.crashed:
                self.crashed = True
                self.crash_reason = f'gear {kind} impact vz {vw[2]:.1f} m/s'
            z = self.zc[i] + vw[:2] * dt
            if kind == 'hard':
                lim = self.mu_slide * N / self.k_tan
                nz = np.linalg.norm(z)
                if nz > lim:
                    z *= lim / max(nz, 1e-9)
            else:
                f = self._wheel_dir(R, steer if kind == 'nose' else 0.0)
                lat = np.array([-f[1], f[0]])
                zl = float(np.clip(z @ lat, -self.mu_lat * N / self.k_tan, self.mu_lat * N / self.k_tan))
                zf = 0.0
                if kind == 'main' and brake > 0.0:
                    lim = self.mu_brake * brake * N / self.k_tan
                    zf = float(np.clip(z @ f, -lim, lim))
                z = zl * lat + zf * f
            self.zc[i] = z
        self.on_ground = touching

    # ------------------------------------------------------------------ dynamics
    def _deriv(self, x, u, wind, rho):
        pos, vel, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        Om, Ah, eps = x[13], x[14], x[15]
        da, de, dr, duty, brake = u
        q = q / np.linalg.norm(q)
        R = q2R(q)
        vb = R.T @ vel                       # inertial velocity, body axes
        wb = R.T @ wind
        va = vb - wb                         # air-relative

        # propulsion
        vp = va + _cross(w, self.rp)
        Vax = max(vp[0], 0.0)
        T, Qp = self.prop(Om, Vax, rho)
        vi = -Vax / 2 + np.sqrt((Vax / 2) ** 2 + max(T, 0.0) / (2 * rho * self.Ap))
        Im, Ib, Vb = self.electrics(duty, Om, Ah)
        Qm = self.Ke * Im
        Qf = self.Ke * self.I0 * np.tanh(Om / 30.0) + self.B_visc * Om
        mdot = rho * self.Ap * (abs(vp[0]) + vi)
        F_prop = np.array([T, -self.kH * mdot * vp[1], -self.kH * mdot * vp[2]])
        h = self.Jr * (w[0] + Om)                          # rotor angular momentum about body x
        hdot = Qm - Qf - Qp

        # aerodynamics
        h_strip = -(pos[2] + self.sr @ R[2, :])
        Fa, Ma, Pa, eps_ss, tau_dw, ae = self.aero(va, w, np.array([da, de, dr]), vi, Vax, Qp, Om, rho,
                                                   h_strip, eps)

        # ground
        if pos[2] > -1.0:
            Fc, Mc, Pc = self.contact(pos, vel, R, w, 0.45 * dr, brake)
        else:
            Fc, Mc, Pc = np.zeros(3), np.zeros(3), 0.0

        F = Fa + F_prop + Fc + self.m * (R.T @ GRAV)
        M = Ma + _cross(self.rp, F_prop) + Mc - Qp * EX

        # Kirchhoff / Newton-Euler with added mass
        Minv = self._Minv(rho)
        A = self.A
        H = self.I @ w + h * EX
        corr = A @ np.concatenate([_cross(w, wb), np.zeros(3)])     # air-relative accel = nu' + w x wind_b
        rhsF = F - self.m * _cross(w, vb) - corr[:3]
        rhsM = M - hdot * EX - _cross(w, H) - corr[3:]
        nud = Minv @ np.concatenate([rhsF, rhsM])
        vbd, wd = nud[:3], nud[3:]

        acc_b = vbd + _cross(w, vb)                      # inertial acceleration, body axes
        xd = np.empty(self.NX)
        xd[0:3] = vel
        xd[3:6] = R @ acc_b
        xd[6:10] = 0.5 * qmul(q, np.array([0.0, *w]))
        xd[10:13] = wd
        xd[13] = hdot / self.Jr - wd[0]
        xd[14] = Ib / 3600.0
        xd[15] = (eps_ss - eps) / tau_dw
        f_spec = acc_b - R.T @ GRAV

        P_prop = float(F_prop @ (vb + _cross(w, self.rp))) - Qp * (w[0] + Om)
        info = dict(T=T, I=Im, Ib=Ib, Vb=Vb, rpm=Om * 60 / (2 * np.pi), Q=Qp, vi=vi, ae=ae,
                    P_aero=Pa, P_prop=P_prop, P_motor=(Qm - Qf) * Om, P_contact=Pc,
                    P_ext=Pa + P_prop + (Qm - Qf) * Om + Pc, P_elec=Vb * Ib)
        return xd, f_spec, info

    def deriv(self, x, u, wind, rho=1.225):
        xd, f, self.info = self._deriv(x, u, wind, rho)
        return xd, f

    def step(self, x, u, wind, dt, rho=1.225):
        k1, f, info = self._deriv(x, u, wind, rho)
        k2, _, _ = self._deriv(x + 0.5 * dt * k1, u, wind, rho)
        k3, _, _ = self._deriv(x + 0.5 * dt * k2, u, wind, rho)
        k4, _, _ = self._deriv(x + dt * k3, u, wind, rho)
        xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[6:10] /= np.linalg.norm(xn[6:10])
        xn[13] = max(xn[13], 0.0)
        self.info = info
        self._update_contacts(xn, dt, 0.45 * u[2], u[4])
        return xn, f, info['T']

    # ------------------------------------------------------------------ diagnostics
    def energy(self, x, wind=np.zeros(3), rho=1.225):
        """Mechanical energy (J): translational + rotational + rotor + entrained air + potential."""
        R = q2R(x[6:10] / np.linalg.norm(x[6:10]))
        vel, w, Om = x[3:6], x[10:13], x[13]
        nur = np.concatenate([R.T @ (vel - wind), w])
        ke_t = 0.5 * self.m * vel @ vel
        ke_r = 0.5 * w @ self.I @ w
        ke_rotor = 0.5 * self.Jr * (w[0] + Om) ** 2
        ke_air = 0.5 * nur @ (rho * self.A1) @ nur
        pe = -self.m * G * x[2]
        return dict(ke_trans=ke_t, ke_rot=ke_r, ke_rotor=ke_rotor, ke_air=ke_air, pe=pe,
                    total=ke_t + ke_r + ke_rotor + ke_air + pe,
                    battery_J=(self.cap_Ah - x[14]) * 3600 * self.cells * 3.7)

    def summary(self):
        A = 1.225 * self.A1
        return (f"m {self.m:.3f} kg  CG(datum) {np.round(self.cg_datum, 3)}  "
                f"I diag {np.round(np.diag(self.I), 4)} Ixz {self.I[0, 2]:.4f}  "
                f"added mass z {A[2, 2]:.3f} kg  added I diag {np.round(np.diag(A)[3:], 4)}")
