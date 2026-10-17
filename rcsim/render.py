"""Software renderer (numpy only, no GL, no GUI toolkit).

* Scene.backdrop(): per-pixel ray-cast sky / clouds / mountains / textured ground with cloud shadows and haze.
  The ground, fields and clouds are baked once into textures (cached on disk) so a frame is just a few
  array look-ups.
* FrameBuilder: projects the aircraft (articulated: ailerons, elevator, rudder, spinning prop), trees,
  buildings, windsock and the ground shadow into painter-ordered screen polygons.
The pygame viewer blits the backdrop and draws the polygons.  World frame is NED (x north, y east, z down).
"""
import os
import warnings
import numpy as np

# Accelerate-backed matmul raises spurious FP-flag warnings on macOS (results are fine)
warnings.filterwarnings('ignore', message='.*encountered in matmul')

SUN_AZ = np.radians(130.0)
SUN_EL = np.radians(38.0)
SUN = np.array([np.cos(SUN_EL) * np.cos(SUN_AZ), np.cos(SUN_EL) * np.sin(SUN_AZ), -np.sin(SUN_EL)], np.float32)
LIGHT = -SUN.astype(float)

HORIZON = np.array([0.80, 0.88, 0.96], np.float32)
ZENITH = np.array([0.14, 0.40, 0.80], np.float32)
HORIZON97 = (HORIZON * 0.97).astype(np.float32)
SUNCOL = np.array([1.0, 0.92, 0.7], np.float32)
FAR_MT = np.array([0.58, 0.65, 0.76], np.float32)
NEAR_MT = np.array([0.38, 0.46, 0.54], np.float32)
SNOW = np.array([0.93, 0.95, 0.98], np.float32)
ASPH = np.array([0.20, 0.20, 0.215], np.float32)
WHITE = np.array([0.93, 0.93, 0.90], np.float32)
RUNWAY_X = (-75.0, 135.0)
RUNWAY_HALF_W = 5.5
SURF_THROW = np.radians(40.0)          # drawn surface deflection at full stick (3D planes throw 35-45 deg)
TEX_N = 1024                           # ground texture: TEX_N texels, GROUND_RES metres each (wraps)
GROUND_RES = 2.0
CLOUD_N = 1024
CLOUD_H = 420.0
CLOUD_TILE = 4500.0                    # metres covered by one cloud texture tile


# ---------------------------------------------------------------------------------------- noise
def _hash(ix, iy):
    h = np.sin(ix * 127.1 + iy * 311.7) * 43758.5453
    return h - np.floor(h)


def vnoise(x, y, period=None):
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    x1, y1 = xi + 1, yi + 1
    if period is not None:
        xi, yi, x1, y1 = xi % period, yi % period, x1 % period, y1 % period
    a, b = _hash(xi, yi), _hash(x1, yi)
    c, d = _hash(xi, y1), _hash(x1, y1)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, octaves=3, period=None):
    s, amp, tot = 0.0, 0.5, 0.0
    for _ in range(octaves):
        s = s + amp * vnoise(x, y, period)
        tot += amp
        x, y, amp = x * 2.0, y * 2.0, amp * 0.5
        if period is not None:
            period *= 2
    return s / tot


# ---------------------------------------------------------------------------------------- camera
class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, -2.0])
        self.f = np.array([1.0, 0.0, 0.0])
        self.r = np.array([0.0, 1.0, 0.0])
        self.u = np.array([0.0, 0.0, -1.0])
        self.fov = np.radians(60.0)               # vertical

    def look(self, pos, target=None, fwd=None, up=(0.0, 0.0, -1.0)):
        self.pos = np.asarray(pos, float)
        f = (np.asarray(target, float) - self.pos) if fwd is None else np.asarray(fwd, float)
        self.f = f / np.linalg.norm(f)
        up = np.asarray(up, float)
        r = np.cross(self.f, up)
        if np.linalg.norm(r) < 1e-6:
            r = np.cross(self.f, np.array([0.0, 1.0, 0.0]))
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)

    def project_px(self, P, W, H):
        """P (...,3) -> pixel x, y and camera depth"""
        d = np.asarray(P, float) - self.pos
        z = d @ self.f
        t = 1.0 / np.tan(self.fov / 2)
        zs = np.maximum(z, 1e-3)
        ndc_x = t * (d @ self.r) / zs / (W / H)
        ndc_y = t * (d @ self.u) / zs
        return (ndc_x + 1) * 0.5 * W, (1 - ndc_y) * 0.5 * H, z


# ---------------------------------------------------------------------------------------- scenery
class Scene:
    def __init__(self, seed=7, cache=True):
        rng = np.random.default_rng(seed)
        pts = []
        for cx, cy, n, spread in [(-160, -190, 40, 70), (230, 150, 46, 80), (120, -330, 36, 60),
                                  (-340, 120, 34, 70), (520, -120, 40, 90), (-60, 330, 30, 50),
                                  (330, 380, 36, 70), (-480, -300, 42, 90), (60, 160, 18, 35), (-110, 90, 14, 30)]:
            for _ in range(n):
                pts.append((cx + rng.normal(0, spread), cy + rng.normal(0, spread)))
        for x in np.arange(-400, 700, 30.0):
            pts.append((x + rng.normal(0, 3), -320 - 8 + rng.normal(0, 1.5)))
            pts.append((x + rng.normal(0, 3), -320 + 9 + rng.normal(0, 1.5)))
        self.trees = []
        for x, y in pts:
            if abs(y) < 16 and RUNWAY_X[0] - 30 < x < RUNWAY_X[1] + 30:
                continue
            if 4 < x < 85 and 12 < y < 44:
                continue
            self.trees.append((x, y, rng.uniform(6.0, 15.0), rng.uniform(0.0, 1.0)))
        self.tree_arr = np.array(self.trees)
        self.boxes = [
            ((38.0, 28.0), (22.0, 14.0, 6.0), (0.78, 0.80, 0.84), (0.55, 0.15, 0.12)),
            ((72.0, 26.0), (5.0, 5.0, 12.0), (0.86, 0.86, 0.82), (0.35, 0.37, 0.42)),
            ((12.0, 30.0), (9.0, 7.0, 4.0), (0.72, 0.62, 0.50), (0.40, 0.18, 0.15)),
            ((-30.0, 40.0), (8.0, 8.0, 7.0), (0.80, 0.78, 0.70), (0.45, 0.25, 0.18)),
        ]
        self.sock_base = np.array([60.0, -16.0])
        bv, bc = [], []
        for (cx, cy), size, wall, roof in self.boxes:
            for q, c in box_faces(cx, cy, size, wall, roof):
                bv.append(q)
                bc.append(_shade(np.cross(q[1] - q[0], q[2] - q[0]), c))
        self.box_verts, self.box_cols = np.array(bv), np.array(bc)
        self._build_textures(cache)
        self.gpack = np.ascontiguousarray(self.gtex).view(np.uint32).reshape(-1)       # RGBA packed, flat
        self.dflat = self.dtex.reshape(-1)
        self.cflat = self.ctex.reshape(-1)
        e = np.linspace(0, np.pi / 2, 512, dtype=np.float32)
        k = 1.0 - np.exp(-e * 3.2)
        self.sky_lut = (HORIZON[None] * (1 - k)[:, None] + ZENITH[None] * k[:, None]).astype(np.float32)
        az = np.linspace(-np.pi, np.pi, 2048, dtype=np.float32)
        self.m1 = (0.012 + 0.030 * (0.5 + 0.5 * np.sin(2.0 * az + 1.0)) ** 2 + 0.035 * np.maximum(np.sin(5 * az + 2.0), 0) ** 2
                   + 0.010 * np.sin(13 * az + 0.4) + 0.012).astype(np.float32)
        self.m2 = (0.006 + 0.022 * (0.5 + 0.5 * np.sin(3.0 * az + 4.0)) ** 2 + 0.012 * np.sin(9 * az + 1.0)
                   + 0.014 + 0.012 * np.maximum(np.sin(7 * az + 0.3), 0)).astype(np.float32)
        self._grid_cache = {}

    def _build_textures(self, cache):
        path = os.path.join(os.path.dirname(__file__), '.texcache_v2.npz')
        if cache and os.path.exists(path):
            try:
                z = np.load(path)
                self.gtex, self.dtex, self.ctex = z['g'], z['d'], z['c']
                return
            except Exception:
                pass
        n = TEX_N
        gt = np.zeros((n, n, 4), np.uint8)
        for r0 in range(0, n, 128):
            ii, jj = np.meshgrid(np.arange(r0, r0 + 128), np.arange(n), indexing='ij')   # ii -> x (north), jj -> y (east)
            x = (ii - n // 2) * GROUND_RES + GROUND_RES / 2
            y = (jj - n // 2) * GROUND_RES + GROUND_RES / 2
            col, stripe = self._ground_proc(x, y)
            gt[r0:r0 + 128, :, :3] = np.clip(col * 255, 0, 255).astype(np.uint8)
            gt[r0:r0 + 128, :, 3] = stripe
        self.gtex = gt
        m = 256
        iy, ix = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
        self.dtex = (fbm(ix / 16.0, iy / 16.0, 4, period=16) * 255).astype(np.uint8)
        m = CLOUD_N
        iy, ix = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
        self.ctex = (fbm(ix / m * 16.0, iy / m * 16.0, 5, period=16) * 255).astype(np.uint8)
        if cache:
            try:
                np.savez(path, g=self.gtex, d=self.dtex, c=self.ctex)
            except Exception:
                pass

    @staticmethod
    def _ground_proc(x, y):
        cx, cy = np.floor(x / 90.0), np.floor(y / 90.0)
        h = _hash(cx, cy)
        h2 = _hash(cx + 17.0, cy - 5.0)
        greens = np.array([[0.30, 0.50, 0.18], [0.40, 0.58, 0.22], [0.52, 0.60, 0.25], [0.62, 0.55, 0.28],
                           [0.26, 0.42, 0.16], [0.70, 0.64, 0.30]])
        gi = np.minimum((h * len(greens)).astype(int), len(greens) - 1)
        col = greens[gi]
        n = fbm(x * 0.05, y * 0.05, 3)
        col = col * (0.88 + 0.24 * n[..., None])
        forest = h > 0.90
        col = np.where(forest[..., None], np.array([0.12, 0.28, 0.10]) * (0.8 + 0.4 * n[..., None]), col)
        stripe = np.where(h2 > 0.45, np.where(h > 0.5, 1, 2), 0).astype(np.uint8)
        stripe = np.where(forest, 0, stripe)
        lk = ((x + 380.0) / 170.0) ** 2 + ((y - 300.0) / 110.0) ** 2
        col = np.where((lk < 1.18)[..., None], np.array([0.78, 0.72, 0.52]), col)
        col = np.where((lk < 1.0)[..., None], np.array([0.12, 0.30, 0.46]) + 0.05 * n[..., None], col)
        stripe = np.where(lk < 1.18, 0, stripe)
        r1 = np.abs(x - 255.0) < 3.4
        r2 = np.abs(y + 320.0) < 3.4
        col = np.where((r1 | r2)[..., None], np.array([0.30, 0.30, 0.31]), col)
        d1 = r1 & (np.abs(x - 255.0) < 0.6) & (np.mod(y, 14.0) < 7.0)
        d2 = r2 & (np.abs(y + 320.0) < 0.6) & (np.mod(x, 14.0) < 7.0)
        col = np.where((d1 | d2)[..., None], np.array([0.85, 0.80, 0.30]), col)
        stripe = np.where(r1 | r2, 0, stripe)
        ap = (x > 4) & (x < 84) & (y > 14) & (y < 44)
        tw = (np.abs(y - 11.0) < 2.2) & (x > 4) & (x < 70)
        col = np.where((ap | tw)[..., None], np.array([0.43, 0.43, 0.44]), col)
        stripe = np.where(ap | tw, 0, stripe)
        return col, stripe

    def _grid(self, W, H, fov):
        key = (W, H, round(float(fov), 5))
        g = self._grid_cache.get(key)
        if g is None:
            tf = np.tan(fov / 2)
            sx = ((np.arange(W) + 0.5) / W * 2 - 1) * tf * (W / H)
            sy = (1 - (np.arange(H) + 0.5) / H * 2) * tf
            X, Y = np.meshgrid(sx.astype(np.float32), sy.astype(np.float32))
            nrm = np.sqrt(1.0 + X * X + Y * Y)
            if len(self._grid_cache) > 12:
                self._grid_cache.clear()
            g = self._grid_cache[key] = (X, Y, nrm)
        return g

    def backdrop(self, cam, W, H, t):
        """uint8 image (H, W, 3)"""
        X, Y, nrm = self._grid(W, H, cam.fov)
        f, r, u = [float(v) for v in cam.f], [float(v) for v in cam.r], [float(v) for v in cam.u]
        px, py, pz = float(cam.pos[0]), float(cam.pos[1]), float(cam.pos[2])
        Dx = f[0] + r[0] * X + u[0] * Y
        Dy = f[1] + r[1] * X + u[1] * Y
        Dz = f[2] + r[2] * X + u[2] * Y
        alt = max(-pz, 0.25)
        img = np.empty((H, W, 3), np.float32)
        flat = img.reshape(-1, 3)
        below = Dz > 1e-4
        si = np.nonzero((~below).ravel())[0]
        sunx, suny, sunz = float(SUN[0]), float(SUN[1]), float(SUN[2])
        if si.size:
            dx, dy, dz, nn = Dx.ravel()[si], Dy.ravel()[si], Dz.ravel()[si], nrm.ravel()[si]
            e = np.arcsin(np.clip(-dz / nn, 0, 1))
            col = self.sky_lut[np.minimum((e * (511 / (np.pi / 2))).astype(np.int32), 511)]
            cs = (dx * sunx + dy * suny + dz * sunz) / nn
            ang2 = 2.0 * (1.0 - np.clip(cs, -1, 1))
            glow = 0.35 * np.exp(-ang2 * (1 / 0.12)) + 1.2 * np.exp(-ang2 * (1 / 0.0016))
            col = col + glow[:, None] * SUNCOL
            if alt < CLOUD_H:
                up = np.maximum(-dz, 1e-3)
                tt = (CLOUD_H - alt) / up
                k = CLOUD_N / CLOUD_TILE
                cxn = ((px + t * 1.2) + tt * dx) * k
                cyn = ((py + t * 0.6) + tt * dy) * k
                n = self.cflat[((cyn.astype(np.int32) & (CLOUD_N - 1)) << 10) | (cxn.astype(np.int32) & (CLOUD_N - 1))] * (1 / 255.0)
                dens = np.clip((n - 0.50) * 5.0, 0, 1) * np.clip(e * (1 / 0.06), 0, 1)
                shade = 0.80 + 0.20 * np.clip((n - 0.5) * 2.2 + 0.4, 0, 1)
                col = col * (1 - 0.95 * dens[:, None]) + (0.95 * dens * shade)[:, None]
            near = e < 0.09
            if near.any():
                az = np.arctan2(dy[near], dx[near])
                ai = np.minimum(((az + np.pi) * (2047 / (2 * np.pi))).astype(np.int32), 2047)
                ee = e[near]
                c = col[near]
                m2 = self.m2[ai]
                rel2 = np.clip(ee / np.maximum(m2, 1e-4), 0, 1)[:, None]
                far = FAR_MT * (0.9 + 0.2 * rel2)
                c = np.where((ee < m2)[:, None], far * 0.85 + HORIZON * 0.15, c)
                m1 = self.m1[ai]
                rel = np.clip(ee / np.maximum(m1, 1e-4), 0, 1)[:, None]
                base = NEAR_MT * (0.78 + 0.34 * rel)
                mc = np.where(rel > 0.74, SNOW, base)
                mc = mc * 0.86 + HORIZON * 0.14 * (1 - rel)
                c = np.where((ee < m1)[:, None], mc, c)
                col[near] = c
            flat[si] = col
        gi = np.nonzero(below.ravel())[0]
        if gi.size:
            dx, dy, dz, nn = Dx.ravel()[gi], Dy.ravel()[gi], Dz.ravel()[gi], nrm.ravel()[gi]
            tt = alt / dz
            gx = px + tt * dx
            gy = py + tt * dy
            dist = tt * nn
            ix = (np.floor(gx * (1 / GROUND_RES)).astype(np.int32) + TEX_N // 2) & (TEX_N - 1)
            iy = (np.floor(gy * (1 / GROUND_RES)).astype(np.int32) + TEX_N // 2) & (TEX_N - 1)
            pk = self.gpack[(ix << 10) | iy]
            col = np.empty((gi.size, 3), np.float32)
            col[:, 0] = (pk & 255)
            col[:, 1] = ((pk >> 8) & 255)
            col[:, 2] = ((pk >> 16) & 255)
            code = (pk >> 24).astype(np.uint8)
            near = np.clip(1.0 - dist * (1 / 500.0), 0.0, 1.0)
            coord = np.where(code == 1, gy, gx)
            stripe = np.sin(coord * 0.9) * (code > 0) * near
            dn = self.dflat[(((gx * 8).astype(np.int32) & 255) << 8) | ((gy * 8).astype(np.int32) & 255)] * (1 / 255.0)
            col *= ((1.0 / 255.0) * (1.0 + 0.07 * stripe + 0.30 * (dn - 0.5) * near))[:, None]
            rwm = (np.abs(gy) < RUNWAY_HALF_W) & (gx > RUNWAY_X[0]) & (gx < RUNWAY_X[1])
            if rwm.any():
                rx, ry = gx[rwm], gy[rwm]
                asph = ASPH[None] * (0.88 + 0.30 * dn[rwm])[:, None]
                center = (np.abs(ry) < 0.2) & (np.mod(rx - RUNWAY_X[0], 14.0) < 7.0)
                edge = np.abs(np.abs(ry) - (RUNWAY_HALF_W - 0.5)) < 0.12
                bar = (np.abs(ry) < RUNWAY_HALF_W - 1.0) & (np.mod(np.floor((ry + 6) / 0.9), 2.0) < 1.0) & \
                    (((rx > RUNWAY_X[0] + 4) & (rx < RUNWAY_X[0] + 9)) | ((rx < RUNWAY_X[1] - 4) & (rx > RUNWAY_X[1] - 9)))
                tdz = (np.abs(ry) > 1.2) & (np.abs(ry) < 3.0) & (np.mod(rx - RUNWAY_X[0] - 22, 40.0) < 4.0) & (rx > RUNWAY_X[0] + 20)
                asph = np.where((center | edge | bar | tdz)[:, None], WHITE, asph)
                col[rwm] = asph
            if alt < CLOUD_H:
                k = CLOUD_N / CLOUD_TILE
                off_x = CLOUD_H * sunx / (-sunz) + t * 1.2
                off_y = CLOUD_H * suny / (-sunz) + t * 0.6
                cn = self.cflat[((((gy + off_y) * k).astype(np.int32) & (CLOUD_N - 1)) << 10) |
                                (((gx + off_x) * k).astype(np.int32) & (CLOUD_N - 1))] * (1 / 255.0)
                col *= (1.0 - 0.28 * np.clip((cn - 0.52) * 5.0, 0, 1))[:, None]
            fog = np.clip(1.0 - np.exp(-dist * (1 / 4200.0)), 0, 1)[:, None]
            col = col * (1 - fog) + HORIZON97[None] * fog
            flat[gi] = col
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8)


# ---------------------------------------------------------------------------------------- aircraft
def _shade(normal_w, base, two_sided=True):
    n = normal_w / (np.linalg.norm(normal_w) + 1e-9)
    c = float(np.dot(n, SUN))
    c = abs(c) if two_sided else max(c, 0.0)
    return np.clip(np.asarray(base) * (0.50 + 0.58 * c), 0, 1)


def _octagon(cx, cy, cz, r, plane='xz', n=10):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if plane == 'xz':
        return np.stack([cx + r * np.cos(a), np.full(n, cy), cz + r * np.sin(a)], 1)
    return np.stack([np.full(n, cx), cy + r * np.cos(a), cz + r * np.sin(a)], 1)


class PlaneModel:
    """Body-frame (x fwd, y right, z down) polygon model of a ~1 m 3D foamie with moving surfaces.
    Geometry follows the physics model: full-span-ish ailerons (28 % chord), big flat stab with a 48 % chord
    elevator, fin+rudder = 25 % of the wing area."""
    STATIONS = [(0.50, 0.028, 0.030, 0.026), (0.44, 0.040, 0.044, 0.036), (0.34, 0.048, 0.052, 0.040),
                (0.12, 0.052, 0.060, 0.044), (-0.12, 0.046, 0.050, 0.040), (-0.34, 0.030, 0.034, 0.028),
                (-0.50, 0.014, 0.016, 0.014), (-0.58, 0.006, 0.008, 0.006)]
    C_BODY = np.array([0.96, 0.96, 0.94])
    C_BELLY = np.array([0.60, 0.62, 0.66])
    C_STRIPE = np.array([0.86, 0.12, 0.14])
    C_WING = np.array([0.98, 0.50, 0.10])
    C_AIL = np.array([1.00, 0.86, 0.12])
    C_ELEV = np.array([1.00, 0.86, 0.12])
    C_FIN = np.array([0.95, 0.95, 0.95])
    C_RUD = np.array([0.88, 0.12, 0.14])

    def __init__(self):
        self._fuse = self._build_fuselage()

    def _build_fuselage(self):
        n = 10
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rings = []
        for x, ry, rt, rb in self.STATIONS:
            ring = []
            for ang in a:
                s = np.sin(ang)
                ring.append([x, ry * np.cos(ang), -s * (rt if s >= 0 else rb)])
            rings.append(np.array(ring))
        faces = []
        for i in range(len(rings) - 1):
            for k in range(n):
                k2 = (k + 1) % n
                q = np.array([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]])
                top = q[:, 2].mean() < -0.004
                col = self.C_BODY if top else self.C_BELLY
                if abs(q[:, 1].mean()) < 0.014 and top and -0.45 < q[:, 0].mean() < 0.3:
                    col = self.C_STRIPE
                faces.append((q, col))
        for k in range(n):
            k2 = (k + 1) % n
            r0 = 0.026
            q = np.array([[0.50, r0 * np.cos(a[k]), r0 * np.sin(a[k])], [0.50, r0 * np.cos(a[k2]), r0 * np.sin(a[k2])],
                          [0.548, 0, 0]])
            faces.append((q, self.C_STRIPE))
        faces.append((np.array([[0.26, -0.018, -0.050], [0.26, 0.018, -0.050], [0.10, 0.026, -0.064], [0.10, -0.026, -0.064]]),
                      np.array([0.10, 0.16, 0.26])))
        return faces

    def build(self, defl, phase, thr, show_prop=True, shape=None):
        """shape = Airframe.morph: ail/ele/rud surface-chord factors, wing chord factor, drag factor
        (a red drag plate under the fuselage, sized from the extra drag area)."""
        sh = shape or {}
        k_ail, k_ele, k_rud = sh.get('ail', 1.0), sh.get('ele', 1.0), sh.get('rud', 1.0)
        k_wing, k_drag = sh.get('wing', 1.0), sh.get('drag', 1.0)
        da, de, dr = defl
        out = [(q, c, 1.0) for q, c in self._fuse]
        A = SURF_THROW
        le = lambda y: 0.115 - 0.05 * abs(y) / 0.5
        te0 = lambda y: -0.095 + 0.03 * abs(y) / 0.5
        chord = lambda y: (le(y) - te0(y)) * k_wing                # wing chord grows aft of the leading edge
        te = lambda y: le(y) - chord(y)
        for side in (-1.0, 1.0):
            ya, yb, yc = 0.05, 0.17, 0.49
            ach = lambda y: min(0.056 * k_ail * k_wing, 0.6 * chord(y))   # aileron chord
            out.append((np.array([[le(ya), side * ya, 0], [le(yb), side * yb, 0], [te(yb), side * yb, 0],
                                  [te(ya), side * ya, 0]]), self.C_WING, 1.0))
            out.append((np.array([[le(yb), side * yb, 0], [le(yc), side * yc, 0], [te(yc) + ach(yc), side * yc, 0],
                                  [te(yb) + ach(yb), side * yb, 0]]), self.C_WING, 1.0))
            delta = (-side * da) * A
            pts = []
            for y, edge in ((yb, 0), (yc, 0), (yc, 1), (yb, 1)):
                hx, ch = te(y) + ach(y), ach(y)
                pts.append([hx, side * y, 0.0] if edge == 0 else
                           [hx - ch * np.cos(delta), side * y, ch * np.sin(delta)])
            out.append((np.array(pts), self.C_AIL, 1.0))
            sy0, sy1 = 0.016, 0.26
            lex = lambda y: -0.415 - 0.025 * y / 0.26
            tex = lambda y: -0.565 + 0.020 * y / 0.26
            chd = lambda y: min(0.062 * k_ele, 0.75 * (lex(y) - tex(y)))   # elevator chord
            out.append((np.array([[lex(sy0), side * sy0, 0], [lex(sy1), side * sy1, 0], [tex(sy1) + chd(sy1), side * sy1, 0],
                                  [tex(sy0) + chd(sy0), side * sy0, 0]]), self.C_WING, 1.0))
            delta_e = -de * A
            pts = []
            for y, edge in ((sy0, 0), (sy1, 0), (sy1, 1), (sy0, 1)):
                hx, ch = tex(y) + chd(y), chd(y)
                pts.append([hx, side * y, 0.0] if edge == 0 else
                           [hx - ch * np.cos(delta_e), side * y, ch * np.sin(delta_e)])
            out.append((np.array(pts), self.C_ELEV, 1.0))
        out.append((np.array([[-0.370, 0, -0.012], [-0.470, 0, -0.225], [-0.510, 0, -0.225], [-0.500, 0, 0.0],
                              [-0.500, 0, 0.075], [-0.430, 0, 0.050]]), self.C_FIN, 1.0))
        crd = 0.072 * k_rud
        delta_r = -dr * A
        c_, s_ = np.cos(delta_r), np.sin(delta_r)
        out.append((np.array([[-0.510, 0, -0.225], [-0.500, 0, 0.075],
                              [-0.500 - crd * 0.85 * c_, crd * 0.85 * s_, 0.075],
                              [-0.510 - crd * c_, crd * s_, -0.225]]), self.C_RUD, 1.0))
        if k_drag > 1.001:                         # airbrake plate on top of the fuselage, area = CdA / 1.2
            half = 0.5 * np.sqrt(0.006 * (k_drag - 1.0) / 1.2)
            out.append((np.array([[-0.02, -half, -0.045], [-0.02, half, -0.045], [-0.02, half, -0.045 - 2 * half],
                                  [-0.02, -half, -0.045 - 2 * half]]), np.array([0.85, 0.10, 0.10]), 1.0))
        wheel = np.array([0.10, 0.10, 0.11])
        for x, y in ((-0.04, -0.16), (-0.04, 0.16), (0.28, 0.0)):
            out.append((np.array([[x - 0.004, y, 0.03], [x + 0.004, y, 0.03], [x + 0.004, y, 0.074], [x - 0.004, y, 0.074]]),
                        np.array([0.55, 0.55, 0.58]), 1.0))
            out.append((_octagon(x, y - 0.006, 0.075, 0.030, 'xz', 12), wheel, 1.0))
            out.append((_octagon(x, y + 0.006, 0.075, 0.030, 'xz', 12), wheel, 1.0))
        if show_prop:
            rpm_f = float(np.clip(thr, 0, 1))
            if rpm_f > 0.2:
                out.append((_octagon(0.52, 0, 0, 0.127, 'yz', 22), np.array([0.28, 0.28, 0.30]), 0.10 + 0.20 * rpm_f))
            for k in range(2):
                ang = phase + k * np.pi
                c, s = np.cos(ang), np.sin(ang)
                ex, ey = np.array([c, s]), np.array([-s, c])
                w = 0.011
                p = lambda rr, wd: [0.512, rr * ex[0] + wd * ey[0], rr * ex[1] + wd * ey[1]]
                out.append((np.array([p(0.02, -w), p(0.127, -w * 0.5), p(0.127, w * 0.5), p(0.02, w)]),
                            np.array([0.06, 0.06, 0.06]), 1.0 if rpm_f < 0.5 else 0.55))
        return out


def box_faces(cx, cy, size, wall, roof):
    sx, sy, sz = size
    x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    z0, z1 = 0.0, -sz
    w = np.array(wall)
    return [
        (np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]), w),
        (np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]), w),
        (np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]), w * 0.88),
        (np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]), w * 0.88),
        (np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]), np.array(roof)),
    ]


def _q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------------------------------- frame builder
class FrameBuilder:
    """Camera control + scene -> painter-ordered polygons: (pts_px (N,2), (r,g,b,a 0-255), kind)."""
    MODES = ['chase', 'observer', 'fpv', 'far-chase', 'tower']

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.scene = Scene()
        self.plane = PlaneModel()
        self.cam = Camera()
        self.mode = 'chase'
        self.cam_yaw = 0.0
        self.shadow_on = True
        self.chase_dist = 3.6
        self.vscale = 1.0

    def cycle_mode(self):
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]

    def setup_camera(self, sim, dt):
        x = sim.x
        pos = x[0:3]
        R = _q2R(x[6:10])
        cam = self.cam
        vel = x[3:6]
        nose = R[:, 0]
        vh = np.hypot(vel[0], vel[1])
        if self.mode in ('chase', 'far-chase'):
            if vh > 4.0:
                tgt, rate = np.arctan2(vel[1], vel[0]), 2.6
            elif np.hypot(nose[0], nose[1]) > 0.6:
                tgt, rate = np.arctan2(nose[1], nose[0]), 0.9
            else:
                tgt, rate = self.cam_yaw, 0.0
            if vh < 1.0 and -pos[2] < 0.5:
                rate = 0.0                              # parked: don't chase heading jitter
            d = (tgt - self.cam_yaw + np.pi) % (2 * np.pi) - np.pi
            self.cam_yaw += d * min(1.0, rate * dt)
            dist, up = (self.chase_dist, 0.75) if self.mode == 'chase' else (9.0, 1.8)
            back = np.array([np.cos(self.cam_yaw), np.sin(self.cam_yaw), 0.0])
            cp = pos - back * dist + np.array([0, 0, -up])
            cp[2] = min(cp[2], -0.35)
            cam.fov = np.radians(60.0)
            cam.look(cp, target=pos + np.array([0, 0, -0.08]))
            self.vscale = 1.5 if self.mode == 'chase' else 3.0
        elif self.mode in ('observer', 'tower'):
            eye = np.array([-55.0, -17.0, -1.7]) if self.mode == 'observer' else np.array([72.0, 26.0, -13.5])
            d = np.linalg.norm(pos - eye)
            cam.fov = np.radians(float(np.clip(np.degrees(2 * np.arctan(13.0 / max(d, 1.0))), 7.0, 65.0)))
            cam.look(eye, target=pos)
            self.vscale = float(np.clip(d / 20.0, 1.0, 8.0))
        else:  # fpv
            tilt = np.radians(8.0)
            f = R @ np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
            cam.fov = np.radians(88.0)
            cam.look(pos + R @ np.array([0.40, 0.0, -0.05]), fwd=f, up=-R[:, 2])
            self.vscale = 1.0
        if cam.pos[2] > -0.25:
            cam.pos = np.array([cam.pos[0], cam.pos[1], -0.25])

    def backdrop(self, sim):
        return self.scene.backdrop(self.cam, self.W, self.H, sim.t)

    def items(self, sim, VW, VH):
        """(shadow_items, object_items) in pixel coordinates of a VWxVH view"""
        cam = self.cam
        sc = self.scene
        out = []

        def fogc(c, d):
            f = 1 - np.exp(-d / 3000.0)
            return np.asarray(c) * (1 - f) + HORIZON * f

        def add(verts_w, col, alpha=1.0, depth=None, shaded=True, kind='o'):
            px, py, z = cam.project_px(verts_w, VW, VH)
            if np.any(z < 0.15):
                return
            if px.max() < -50 or px.min() > VW + 50 or py.max() < -50 or py.min() > VH + 50:
                return
            c = _shade(np.cross(verts_w[1] - verts_w[0], verts_w[2] - verts_w[0]), col) if shaded and len(verts_w) >= 3 \
                else np.asarray(col)
            out.append((float(z.mean()) if depth is None else depth, np.stack([px, py], 1),
                        (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), int(alpha * 255)), kind))

        # trees: projected in one go, drawn as upright billboards (screen-space polygons, LOD by distance)
        ta = sc.tree_arr
        P = np.stack([ta[:, 0], ta[:, 1], np.zeros(len(ta))], 1)
        rel = P - cam.pos
        zc = rel @ cam.f
        dxy = np.hypot(rel[:, 0], rel[:, 1])
        idx = np.nonzero((zc > 0.8) & (dxy < 2000))[0]
        if idx.size > 200:
            idx = idx[np.argsort(dxy[idx])[:200]]
        if idx.size:
            bx, by, bz = cam.project_px(P[idx], VW, VH)
            k = (VH * 0.5) / np.tan(cam.fov / 2) / np.maximum(bz, 0.5)
            fg = 1 - np.exp(-dxy[idx] / 3000.0)
            hz = HORIZON.astype(float)
            for j in range(idx.size):
                i = idx[j]
                th, tv = ta[i, 2], ta[i, 3]
                x0, y0, kk, d, f = float(bx[j]), float(by[j]), float(k[j]), float(dxy[i]), float(fg[j])
                if x0 < -80 or x0 > VW + 80 or y0 < -20 or y0 - th * kk > VH + 20:
                    continue
                tw = th * 0.30 * kk
                H_ = th * kk
                if tv < 0.66:
                    g = (0.10 + 0.0 * tv, 0.32, 0.14)
                    sc_ = 0.8 + 0.4 * tv
                else:
                    g = (0.30, 0.46, 0.15)
                    sc_ = 0.8 + 0.4 * tv
                if d > 300.0:                                   # far: one silhouette polygon
                    c = ((g[0] * sc_) * (1 - f) + hz[0] * f, (g[1] * sc_) * (1 - f) + hz[1] * f, (g[2] * sc_) * (1 - f) + hz[2] * f)
                    col = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)
                    if tv < 0.66:
                        pts = [(int(x0 - tw * 0.9), int(y0 - 0.18 * H_)), (int(x0), int(y0 - H_)), (int(x0 + tw * 0.9), int(y0 - 0.18 * H_))]
                    else:
                        pts = [(int(x0 - tw), int(y0 - 0.45 * H_)), (int(x0), int(y0 - H_)), (int(x0 + tw), int(y0 - 0.45 * H_)),
                               (int(x0), int(y0 - 0.2 * H_))]
                    out.append((d, pts, col, 'o'))
                    continue

                def col(m, g=g, sc_=sc_, f=f):
                    return (int(((g[0] * sc_ * m) * (1 - f) + hz[0] * f) * 255), int(((g[1] * sc_ * m) * (1 - f) + hz[1] * f) * 255),
                            int(((g[2] * sc_ * m) * (1 - f) + hz[2] * f) * 255), 255)
                tk = ((0.30 * (1 - f) + hz[0] * f), (0.20 * (1 - f) + hz[1] * f), (0.12 * (1 - f) + hz[2] * f))
                out.append((d, [(int(x0 - 0.3 * kk), int(y0)), (int(x0 + 0.3 * kk), int(y0)), (int(x0 + 0.22 * kk), int(y0 - 0.4 * H_)),
                                (int(x0 - 0.22 * kk), int(y0 - 0.4 * H_))], (int(tk[0] * 255), int(tk[1] * 255), int(tk[2] * 255), 255), 'o'))
                if tv < 0.66:
                    for z0, z1, w in ((0.18, 0.62, 1.0), (0.45, 1.0, 0.72)):
                        my = int(y0 - H_ * z0)
                        tip = (int(x0), int(y0 - H_ * z1))
                        out.append((d - 0.01, [(int(x0), my), (int(x0 + tw * w), my), tip], col(1.18), 'o'))
                        out.append((d - 0.01, [(int(x0), my), tip, (int(x0 - tw * w), my)], col(0.78), 'o'))
                else:
                    cy_ = y0 - 0.68 * H_
                    ang = np.linspace(0, 2 * np.pi, 9, endpoint=False)
                    ca, sa = np.cos(ang), np.sin(ang)
                    out.append((d, list(zip((x0 + tw * 1.15 * ca).astype(int).tolist(), (cy_ + 0.34 * H_ * sa).astype(int).tolist())), col(0.85), 'o'))
                    out.append((d - 0.01, list(zip((x0 - 0.2 * tw + tw * 0.8 * ca).astype(int).tolist(),
                                                   (cy_ - 0.07 * H_ + 0.34 * H_ * sa).astype(int).tolist())), col(1.15), 'o'))
        # buildings: constant face normals -> pre-shaded colours, one batched projection
        bv, bc = sc.box_verts, sc.box_cols
        bpx, bpy, bpz = cam.project_px(bv.reshape(-1, 3), VW, VH)
        bpx, bpy, bpz = bpx.reshape(-1, 4), bpy.reshape(-1, 4), bpz.reshape(-1, 4)
        for i in range(len(bv)):
            if bpz[i].min() < 0.15:
                continue
            d = float(bpz[i].mean())
            f = 1 - np.exp(-d / 3000.0)
            c = bc[i] * (1 - f) + HORIZON * f
            out.append((d, list(zip(bpx[i].astype(int).tolist(), bpy[i].astype(int).tolist())),
                        (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255), 'o'))
        self._windsock(sim, add)
        x = sim.x
        R = _q2R(x[6:10])
        pos = x[0:3]
        S = self.vscale
        faces = self.plane.build((sim.servo[0], sim.servo[1], sim.servo[2]), sim.rotor_phase, sim.duty,
                                 shape=getattr(sim.af, 'morph', None),
                                 show_prop=self.mode != 'fpv')
        if self.mode == 'fpv':                       # camera sits at the nose: hide the spinner
            faces = [f for f in faces if f[0][:, 0].min() < 0.49]
        lens = np.array([len(q) for q, _, _ in faces])
        off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        allv = np.concatenate([q for q, _, _ in faces]) * S
        W3 = pos + allv @ R.T
        px, py, pz = cam.project_px(W3, VW, VH)
        gsh = W3 + (-W3[:, 2] / LIGHT[2])[:, None] * LIGHT[None, :]
        gsh[:, 2] = -0.02
        sx_, sy_, sz_ = cam.project_px(gsh, VW, VH)
        a_, b_, c_ = W3[off], W3[off + 1], W3[off + 2]
        e1, e2 = b_ - a_, c_ - a_
        nrm = np.stack([e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1], e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
                        e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]], 1)
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        lit = np.abs(nrm @ SUN.astype(float))
        base = np.array([c for _, c, _ in faces])
        cols = np.clip(base * (0.50 + 0.58 * lit)[:, None], 0, 1)
        cols = (cols * 255).astype(int)
        zmin = np.minimum.reduceat(pz, off)
        zmean = np.add.reduceat(pz, off) / lens
        szmin = np.minimum.reduceat(sz_, off)
        shadows = []
        for i, (q, c, al) in enumerate(faces):
            if zmin[i] < 0.08:
                continue
            o, n = int(off[i]), int(lens[i])
            out.append((float(zmean[i]), list(zip(px[o:o + n].astype(int).tolist(), py[o:o + n].astype(int).tolist())),
                        (int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2]), int(al * 255)), 'p'))
            if self.shadow_on and al > 0.5 and szmin[i] > 0.2:
                shadows.append(list(zip(sx_[o:o + n].astype(int).tolist(), sy_[o:o + n].astype(int).tolist())))
        out.sort(key=lambda it: -it[0])
        return [(s, (10, 16, 8, 62), 's') for s in shadows], [(it[1], it[2], it[3]) for it in out]

    def _windsock(self, sim, add):
        sc = self.scene
        cam = self.cam
        b = np.array([sc.sock_base[0], sc.sock_base[1], 0.0])
        top = b + np.array([0, 0, -6.0])
        w = sim.atm.mean(6.0)
        sp = np.hypot(w[0], w[1])
        az = np.arctan2(w[1], w[0]) if sp > 0.05 else 0.0
        droop = np.clip(1.0 - sp / 7.0, 0.0, 1.0)
        ax_ = np.array([np.cos(az), np.sin(az), 0.0])
        rel = top - cam.pos
        d = np.linalg.norm(rel)
        if d > 1.0 and rel @ cam.f > 0.5:
            side = np.array([-rel[1], rel[0], 0.0])
            side /= (np.linalg.norm(side) + 1e-9)
            add(np.array([b - side * 0.1, b + side * 0.1, top + side * 0.06, top - side * 0.06]),
                np.array([0.85, 0.85, 0.85]), depth=d, shaded=False)
            L = 3.2
            tip = ax_ * (1 - droop) + np.array([0, 0, 1.0]) * droop
            tip /= np.linalg.norm(tip)
            seg = 6
            for i in range(seg):
                p0 = top + tip * L * i / seg
                p1 = top + tip * L * (i + 1) / seg
                r0, r1 = 0.45 * (1 - 0.6 * i / seg), 0.45 * (1 - 0.6 * (i + 1) / seg)
                perp = np.cross(tip, np.array([0, 0, 1.0]))
                perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else np.array([0, 1.0, 0])
                q = np.array([p0 + perp * r0, p1 + perp * r1, p1 - perp * r1, p0 - perp * r0])
                col = np.array([0.95, 0.35, 0.10]) if i % 2 == 0 else np.array([0.97, 0.97, 0.97])
                add(q, col, depth=np.linalg.norm(p0 - cam.pos), shaded=False)
