#!/usr/bin/env python3
"""
MolGame — Pixel Edition
Real protein MD + retro pixel-art HUD.  WASD steer, mouse look, scroll zoom.
Usage:  python molgame.py [--pdb 1UBQ]
"""

import sys, math, os, time, argparse
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import openmm as mm
from openmm import unit
import openmm.app as app
from pdbfixer import PDBFixer


# ── Configuration ───────────────────────────────────────────
W, H            = 1280, 720
FPS_CAP         = 60
MD_STEPS        = 8
DT              = 0.004
FORCE_MAG       = 500.0
FRICTION        = 5.0
MOUSE_SENS      = 0.15
SCROLL_SENS     = 0.5
TEMPERATURE     = 300.0
RESTRAINT_K     = 0.0
CONTACT_CUT     = 0.5

CPK = {"C": (0.32, 0.32, 0.32), "N": (0.14, 0.20, 0.65),
       "O": (0.65, 0.14, 0.14), "S": (0.65, 0.60, 0.14)}
VDW_RENDER = {"C": 0.09, "N": 0.08, "O": 0.08, "S": 0.10}
VDW_REAL   = {"C": 0.170, "N": 0.155, "O": 0.152, "S": 0.180}

# ── Pixel palette (Contra / NES inspired) ──────────────────
PX = {
    "bg":      (6, 6, 16, 215),
    "border":  (0, 210, 255),
    "border2": (0, 100, 140),
    "title":   (255, 185, 20),
    "stage":   (0, 255, 210),
    "text":    (210, 210, 225),
    "dim":     (80, 80, 110),
    "bar_lo":  (0, 200, 90),
    "bar_mid": (230, 210, 0),
    "bar_hi":  (255, 60, 30),
    "bar_bg":  (26, 26, 38),
    "key_on":  (0, 210, 80),
    "key_off": (32, 32, 48),
    "ok":      (50, 255, 130),
    "warn":    (255, 255, 60),
    "alert":   (255, 50, 50),
    "sep":     (36, 52, 72),
}


# ── Molecular Surface (marching cubes) ─────────────────────
def compute_surface(atom_pos, elem_list, probe_r=0.14, spacing=0.08):
    from skimage.measure import marching_cubes
    radii = np.array([VDW_REAL.get(e, 0.15) + probe_r for e in elem_list])
    margin = 0.3
    lo = atom_pos.min(axis=0) - margin
    hi = atom_pos.max(axis=0) + margin
    shape = ((hi - lo) / spacing).astype(int) + 1
    density = np.zeros(shape, dtype=np.float32)
    for p, r in zip(atom_pos, radii):
        idx = ((p - lo) / spacing).astype(int)
        rg = int(r / spacing) + 2
        s = tuple(slice(max(0, idx[i] - rg), min(shape[i], idx[i] + rg + 1)) for i in range(3))
        ii, jj, kk = np.mgrid[s]
        coords = lo + np.stack([ii, jj, kk], axis=-1).astype(np.float32) * spacing
        dist2 = np.sum((coords - p) ** 2, axis=-1)
        density[s] = np.maximum(density[s], np.exp(-dist2 / (0.6 * r) ** 2))
    verts, faces, normals, _ = marching_cubes(density, 0.5, spacing=(spacing,) * 3)
    verts = verts + lo
    return verts.astype(np.float32), faces, normals.astype(np.float32)


def build_surface_dl(verts, faces, normals):
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    glColor3f(0.28, 0.42, 0.58)
    glBegin(GL_TRIANGLES)
    for f in faces:
        for i in f:
            glNormal3fv(normals[i])
            glVertex3fv(verts[i])
    glEnd()
    glEndList()
    return dl


# ── PDB helpers ─────────────────────────────────────────────
def download_pdb(pdb_id, dest):
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading {pdb_id.upper()} from RCSB …")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"ERROR: Could not download {pdb_id}: {e}")
        sys.exit(1)


def read_pdb_title(path):
    title = ""
    with open(path) as f:
        for line in f:
            if line.startswith("TITLE"):
                title += line[10:].strip() + " "
            elif line.startswith("ATOM"):
                break
    title = title.strip()
    return title[:32] + ".." if len(title) > 34 else title


# ── Prepare molecular system ───────────────────────────────
def prepare(pdb_path):
    t0 = time.time()
    print("[1/6] Fixing PDB …")
    fixer = PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(False)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    print("[2/6] Solvating with TIP3P …")
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(ff, model="tip3p", padding=0.8 * unit.nanometers)

    prot_heavy, prot_elem, water_o, ca_idx = [], [], [], []
    for atom in modeller.topology.atoms():
        if atom.residue.name in ("HOH", "WAT"):
            if atom.element.symbol == "O":
                water_o.append(atom.index)
        elif atom.element.symbol != "H":
            prot_heavy.append(atom.index)
            prot_elem.append(atom.element.symbol)
            if atom.name == "CA":
                ca_idx.append(atom.index)

    pos = np.array(modeller.positions.value_in_unit(unit.nanometers))
    prot_pos = pos[prot_heavy]
    prot_center = prot_pos.mean(axis=0)

    dx = prot_pos[:, 0] - prot_center[0]
    surf = prot_pos[np.argmax(dx)]
    direction = surf - prot_center
    direction /= np.linalg.norm(direction)
    lig_center = surf + direction * 1.0

    print("[3/6] Clearing water overlap …")
    to_delete = []
    for res in modeller.topology.residues():
        if res.name in ("HOH", "WAT"):
            for atom in res.atoms():
                if atom.element.symbol == "O":
                    if np.linalg.norm(pos[atom.index] - lig_center) < 0.4:
                        to_delete.append(res)
                    break
    if to_delete:
        modeller.delete(to_delete)
        pos = np.array(modeller.positions.value_in_unit(unit.nanometers))
        prot_heavy, prot_elem, water_o, ca_idx = [], [], [], []
        for atom in modeller.topology.atoms():
            if atom.residue.name in ("HOH", "WAT"):
                if atom.element.symbol == "O":
                    water_o.append(atom.index)
            elif atom.element.symbol != "H":
                prot_heavy.append(atom.index)
                prot_elem.append(atom.element.symbol)
                if atom.name == "CA":
                    ca_idx.append(atom.index)
        print(f"     Removed {len(to_delete)} water molecules")

    heavy_set = set(prot_heavy)
    heavy_bonds = []
    for bond in modeller.topology.bonds():
        a, b = bond[0].index, bond[1].index
        if a in heavy_set and b in heavy_set:
            heavy_bonds.append((a, b))

    print(f"     Protein heavy atoms: {len(prot_heavy)},  Bonds: {len(heavy_bonds)},  Water: {len(water_o)}")

    print("[4/6] Building force field …")
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometers, constraints=app.HBonds)

    rst = mm.CustomExternalForce("rst_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rst.addGlobalParameter("rst_k", RESTRAINT_K)
    rst.addPerParticleParameter("x0")
    rst.addPerParticleParameter("y0")
    rst.addPerParticleParameter("z0")
    for idx in ca_idx:
        rst.addParticle(idx, pos[idx].tolist())
    system.addForce(rst)

    lig_idx = system.addParticle(40.0)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    nb.addParticle(0.0, 0.40, 6.0)

    pf = mm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    pf.addGlobalParameter("fx", 0.0)
    pf.addGlobalParameter("fy", 0.0)
    pf.addGlobalParameter("fz", 0.0)
    pf.addParticle(lig_idx, [])
    system.addForce(pf)

    all_pos = np.vstack([pos, [lig_center]])
    ligand = np.array([lig_idx])
    print(f"     Total particles: {system.getNumParticles()}")

    print("[5/6] Computing molecular surface …")
    surface_data = compute_surface(pos[prot_heavy], prot_elem)

    print("[6/6] Creating simulation (OpenCL) …")
    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE * unit.kelvin, FRICTION / unit.picosecond, DT * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("OpenCL")
    ctx = mm.Context(system, integrator, platform)
    ctx.setPositions(all_pos * unit.nanometers)
    print("      Minimising …")
    mm.LocalEnergyMinimizer.minimize(ctx, tolerance=10.0, maxIterations=500)
    ctx.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin)
    print("      Equilibrating …")
    integrator.step(500)
    box = ctx.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    box_origin = pos.min(axis=0)

    print(f"      Box: {box[0][0]:.2f} x {box[1][1]:.2f} x {box[2][2]:.2f} nm")
    print(f"      Ready in {time.time()-t0:.1f}s\n")
    return (ctx, integrator,
            np.array(prot_heavy), np.array(prot_elem),
            np.array(water_o), ligand, prot_center, surface_data,
            box_origin, np.diag(box), heavy_bonds)


# ── Camera ──────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.yaw, self.pitch, self.dist = 0.0, 20.0, 6.0
        self.target = np.zeros(3)

    def rotate(self, dx, dy):
        self.yaw += dx * MOUSE_SENS
        self.pitch = np.clip(self.pitch + dy * MOUSE_SENS, -85, 85)

    def zoom(self, d):
        self.dist = np.clip(self.dist - d * SCROLL_SENS, 2.0, 25.0)

    def track(self, p, s=0.07):
        self.target += (p - self.target) * s

    def apply(self):
        glLoadIdentity()
        yr, pr = math.radians(self.yaw), math.radians(self.pitch)
        eye = self.target + self.dist * np.array([
            math.sin(yr) * math.cos(pr), math.sin(pr), math.cos(yr) * math.cos(pr)])
        gluLookAt(*eye, *self.target, 0, 1, 0)

    def forward(self):
        yr, pr = math.radians(self.yaw), math.radians(self.pitch)
        return np.array([-math.sin(yr)*math.cos(pr), -math.sin(pr), -math.cos(yr)*math.cos(pr)])

    def right(self):
        yr = math.radians(self.yaw)
        return np.array([math.cos(yr), 0, -math.sin(yr)])


# ── Rendering ───────────────────────────────────────────────
_sdl = None
_quad = None
_surf_dl = None
_hud_font = None
_hud_font_sm = None


def _txt(text, color, big=True):
    global _hud_font, _hud_font_sm
    if _hud_font is None:
        _hud_font = pygame.font.SysFont("menlo", 14)
        _hud_font_sm = pygame.font.SysFont("menlo", 11)
    font = _hud_font if big else _hud_font_sm
    return font.render(text, True, color)


def gl_init(aw, ah):
    global _sdl, _quad
    glClearColor(0.015, 0.015, 0.04, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [5, 10, 7, 0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.15, 0.15, 0.17, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.55, 0.53, 0.50, 1])
    glLightfv(GL_LIGHT1, GL_POSITION, [-4, 4, -6, 0])
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.18, 0.20, 0.24, 1])
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(50, aw / ah, 0.05, 200)
    glMatrixMode(GL_MODELVIEW)
    _quad = gluNewQuadric(); gluQuadricNormals(_quad, GLU_SMOOTH)
    _sdl = glGenLists(1)
    glNewList(_sdl, GL_COMPILE); gluSphere(_quad, 1.0, 10, 5); glEndList()


def draw_protein_atoms(pos, prot_heavy, prot_elem):
    by_elem = {}
    for idx, el in zip(prot_heavy, prot_elem):
        by_elem.setdefault(el, []).append(idx)
    for el, indices in by_elem.items():
        col = CPK.get(el, (0.35, 0.35, 0.35))
        r = VDW_RENDER.get(el, 0.07)
        glColor3f(*col)
        for idx in indices:
            glPushMatrix()
            p = pos[idx]
            glTranslatef(float(p[0]), float(p[1]), float(p[2]))
            glScalef(r, r, r)
            glCallList(_sdl)
            glPopMatrix()


def draw_protein_sticks(pos, bond_a, bond_b, bond_colors):
    n = len(bond_a)
    verts = np.empty((n * 2, 3), dtype=np.float32)
    verts[0::2] = pos[bond_a]
    verts[1::2] = pos[bond_b]
    glDisable(GL_LIGHTING)
    glLineWidth(2.0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glColorPointer(3, GL_FLOAT, 0, bond_colors)
    glDrawArrays(GL_LINES, 0, n * 2)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glEnable(GL_LIGHTING)


def draw_protein_surface():
    if _surf_dl is not None:
        glCallList(_surf_dl)


WATER_VIS_CUT = 1.5

def draw_water(pos, water_o, lig_pos):
    wp_all = pos[water_o]
    dists = np.linalg.norm(wp_all - lig_pos, axis=1)
    nearby = wp_all[dists < WATER_VIS_CUT].astype(np.float32)
    if len(nearby) == 0:
        return
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.40, 0.60, 0.92, 0.30)
    glPointSize(2.0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, nearby)
    glDrawArrays(GL_POINTS, 0, len(nearby))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisable(GL_BLEND); glEnable(GL_LIGHTING)


def draw_ligand(pos, lig):
    p = pos[lig[0]]
    glPushMatrix()
    glTranslatef(float(p[0]), float(p[1]), float(p[2]))
    glColor3f(0.15, 0.95, 0.30)
    gluSphere(_quad, 0.18, 16, 8)
    glPopMatrix()
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.3, 1.0, 0.5, 0.3)
    glPushMatrix()
    glTranslatef(float(p[0]), float(p[1]), float(p[2]))
    gluSphere(_quad, 0.25, 12, 6)
    glPopMatrix()
    glDisable(GL_BLEND); glEnable(GL_LIGHTING)


def draw_grid(origin, lengths, step=0.5):
    y = origin[1]
    x0, z0 = origin[0], origin[2]
    x1, z1 = x0 + lengths[0], z0 + lengths[2]
    glDisable(GL_LIGHTING)
    glColor3f(0.07, 0.14, 0.20)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    x = x0
    while x <= x1 + 1e-6:
        glVertex3f(x, y, z0); glVertex3f(x, y, z1)
        x += step
    z = z0
    while z <= z1 + 1e-6:
        glVertex3f(x0, y, z); glVertex3f(x1, y, z)
        z += step
    glEnd()
    glEnable(GL_LIGHTING)


def draw_box(origin, lengths):
    o = origin
    L = lengths
    glDisable(GL_LIGHTING)
    glColor3f(0.15, 0.35, 0.45)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    for dx, dy in [(0,0),(L[0],0),(L[0],L[1]),(0,L[1])]:
        glVertex3f(o[0]+dx, o[1]+dy, o[2])
        glVertex3f(o[0]+dx, o[1]+dy, o[2]+L[2])
    for dz in [0, L[2]]:
        glVertex3f(o[0],     o[1],     o[2]+dz)
        glVertex3f(o[0]+L[0],o[1],     o[2]+dz)
        glVertex3f(o[0]+L[0],o[1],     o[2]+dz)
        glVertex3f(o[0]+L[0],o[1]+L[1],o[2]+dz)
        glVertex3f(o[0]+L[0],o[1]+L[1],o[2]+dz)
        glVertex3f(o[0],     o[1]+L[1],o[2]+dz)
        glVertex3f(o[0],     o[1]+L[1],o[2]+dz)
        glVertex3f(o[0],     o[1],     o[2]+dz)
    glEnd()
    glEnable(GL_LIGHTING)


def draw_axes(aw, ah, cam):
    sz = int(min(aw, ah) * 0.10)
    margin = 10
    glViewport(aw - sz - margin, margin, sz, sz)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    gluPerspective(50, 1.0, 0.1, 10)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    yr, pr = math.radians(cam.yaw), math.radians(cam.pitch)
    eye = 3.0 * np.array([
        math.sin(yr) * math.cos(pr), math.sin(pr), math.cos(yr) * math.cos(pr)])
    gluLookAt(*eye, 0, 0, 0, 0, 1, 0)
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glLineWidth(2.5)
    for (dx, dy, dz), col in [((1,0,0),(0.9,0.2,0.2)),
                               ((0,1,0),(0.2,0.9,0.2)),
                               ((0,0,1),(0.3,0.4,0.9))]:
        glColor3f(*col); glBegin(GL_LINES)
        glVertex3f(0,0,0); glVertex3f(dx,dy,dz); glEnd()
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()
    glViewport(0, 0, aw, ah)


def draw_crosshair(aw, ah, size=12, gap=4):
    cx, cy = aw // 2, ah // 2
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, aw, 0, ah, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1.0, 1.0, 1.0, 0.5)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glVertex2i(cx - size, cy); glVertex2i(cx - gap, cy)
    glVertex2i(cx + gap, cy);  glVertex2i(cx + size, cy)
    glVertex2i(cx, cy - size); glVertex2i(cx, cy - gap)
    glVertex2i(cx, cy + gap);  glVertex2i(cx, cy + size)
    glEnd()
    glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()


# ── Scanline CRT overlay ───────────────────────────────────
def draw_scanlines(aw, ah):
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, aw, 0, ah, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.0, 0.0, 0.0, 0.055)
    glBegin(GL_LINES)
    for y in range(0, ah, 3):
        glVertex2i(0, y); glVertex2i(aw, y)
    glEnd()
    glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()


# ── HUD ────────────────────────────────────────────────────
def draw_hud(aw, ah, pdb_id, pdb_title, pe, temp, contacts,
             hi_score, fps, active_keys, view_name, frame):
    pad = 8
    lh = 20
    bw = 310

    if contacts >= 12:
        flash = (frame // 8) % 2
        st_txt, st_col = "IN POCKET!", PX["ok"] if flash else PX["title"]
    elif contacts >= 4:
        st_txt, st_col = "CONTACT!", PX["warn"]
    else:
        st_txt, st_col = "EXPLORING", PX["dim"]

    bh = 260
    surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
    surf.fill((6, 6, 16, 180))
    pygame.draw.rect(surf, PX["border2"], (0, 0, bw, bh), 1)

    y = pad

    # title + stage
    surf.blit(_txt(f"MOLGAME  {pdb_id}", PX["title"]), (pad, y))
    y += lh

    # proximity bar
    bar_x = pad
    seg_w, seg_h = 10, 12
    max_seg = 15
    fill = min(max_seg, contacts)
    for i in range(max_seg):
        sx = bar_x + i * (seg_w + 2)
        if i < fill:
            t = i / max(1, max_seg - 1)
            c = PX["bar_lo"] if t < 0.4 else PX["bar_mid"] if t < 0.7 else PX["bar_hi"]
        else:
            c = PX["bar_bg"]
        pygame.draw.rect(surf, c, (sx, y, seg_w, seg_h))
    surf.blit(_txt(f" {contacts}", PX["text"], big=False),
              (bar_x + max_seg * (seg_w + 2) + 2, y - 1))
    y += seg_h + 6

    # status + hi-score
    surf.blit(_txt(st_txt, st_col), (pad, y))
    hi_t = _txt(f"HI {hi_score}", PX["ok"], big=False)
    surf.blit(hi_t, (bw - pad - hi_t.get_width(), y + 2))
    y += lh + 2

    pygame.draw.line(surf, PX["sep"], (pad, y), (bw - pad, y))
    y += 5

    # info
    surf.blit(_txt(f"PE {pe:10.0f} kJ/mol", PX["dim"], big=False), (pad, y))
    y += lh - 4
    surf.blit(_txt(f"Temp {temp:5.0f}K  FPS {fps:3.0f}", PX["dim"], big=False), (pad, y))
    y += lh - 4
    vn = view_name
    surf.blit(_txt(f"View: {vn} [V]", PX["dim"], big=False), (pad, y))
    y += lh - 2

    pygame.draw.line(surf, PX["sep"], (pad, y), (bw - pad, y))
    y += 5

    # key indicators
    key_labels = ["W", "A", "S", "D", "SPC", "SHF"]
    box_w_map = {"W": 26, "A": 26, "S": 26, "D": 26, "SPC": 38, "SHF": 38}
    box_h = 20
    kx = pad
    for label, active in zip(key_labels, active_keys):
        kw = box_w_map[label]
        bg = PX["key_on"] if active else PX["key_off"]
        brd = PX["border"] if active else PX["sep"]
        fg = (0, 0, 0) if active else PX["dim"]
        pygame.draw.rect(surf, bg, (kx, y, kw, box_h))
        pygame.draw.rect(surf, brd, (kx, y, kw, box_h), 1)
        kt = _txt(label, fg, big=False)
        surf.blit(kt, (kx + (kw - kt.get_width()) // 2,
                        y + (box_h - kt.get_height()) // 2))
        kx += kw + 3
    y += box_h + 4

    surf.blit(_txt("Mouse:look  Scroll:zoom  ESC:quit", PX["dim"], big=False),
              (pad, y))
    y += 16

    # crop and blit
    final_h = y + 4
    out = surf.subsurface((0, 0, bw, final_h)).copy()
    pygame.draw.rect(out, PX["border2"], (0, 0, bw, final_h), 1)

    data = pygame.image.tostring(out, "RGBA", True)
    sw, sh = out.get_size()
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, aw, 0, ah, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glRasterPos2i(10, ah - sh - 10)
    glDrawPixels(sw, sh, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()


# ── Pause menu ─────────────────────────────────────────────
def draw_pause_menu(aw, ah, params, sel):
    pw, ph = 360, len(params) * 28 + 70
    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((6, 6, 20, 220))
    pygame.draw.rect(surf, (0, 180, 220), (0, 0, pw, ph), 2)

    y = 10
    title = _txt("PAUSED  [P] resume", (255, 200, 50))
    surf.blit(title, ((pw - title.get_width()) // 2, y))
    y += 30
    pygame.draw.line(surf, (40, 60, 80), (10, y), (pw - 10, y))
    y += 8

    for i, p in enumerate(params):
        selected = (i == sel)
        col = (0, 255, 200) if selected else (160, 160, 180)
        arrow = "> " if selected else "  "
        v = p["val"]
        if "choices" in p:
            vstr = p["choices"][int(v)]
        elif v == int(v):
            vstr = f"{int(v)}"
        else:
            vstr = f"{v:.1f}"
        line = f"{arrow}{p['name']:14s} {vstr:>8s} {p['unit']}"
        surf.blit(_txt(line, col, big=False), (10, y))
        y += 28

    surf.blit(_txt("  Left/Right to adjust", (100, 100, 130), big=False),
              (10, y))

    data = pygame.image.tostring(surf, "RGBA", True)
    sw, sh = surf.get_size()
    cx, cy = (aw - sw) // 2, (ah - sh) // 2
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, aw, 0, ah, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glRasterPos2i(cx, cy)
    glDrawPixels(sw, sh, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()


# ── Main ────────────────────────────────────────────────────
def main():
    global _surf_dl

    parser = argparse.ArgumentParser(description="MolGame — Molecular Dynamics Game")
    parser.add_argument("--pdb", default="1UBQ", help="PDB ID (default: 1UBQ)")
    args = parser.parse_args()
    pdb_id = args.pdb.upper()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_file = os.path.join(base_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(pdb_file):
        download_pdb(pdb_id, pdb_file)

    pdb_title = read_pdb_title(pdb_file)
    print(f"Protein: {pdb_id}  {pdb_title}")

    (ctx, integrator, prot_heavy, prot_elem,
     water_o, ligand, prot_center, surface_data,
     box_origin, box_lengths, heavy_bonds) = prepare(pdb_file)
    elem_map = {int(i): str(e) for i, e in zip(prot_heavy, prot_elem)}
    bond_a = np.array([a for a, b in heavy_bonds], dtype=np.int32)
    bond_b = np.array([b for a, b in heavy_bonds], dtype=np.int32)
    bond_colors = np.empty((len(heavy_bonds) * 2, 3), dtype=np.float32)
    for i, (a, b) in enumerate(heavy_bonds):
        bond_colors[i * 2] = CPK.get(elem_map[a], (0.35, 0.35, 0.35))
        bond_colors[i * 2 + 1] = CPK.get(elem_map[b], (0.35, 0.35, 0.35))

    # ── Pygame + OpenGL ──
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"MolGame — {pdb_id}")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    vp = glGetIntegerv(GL_VIEWPORT)
    aw, ah = int(vp[2]), int(vp[3])
    gl_init(aw, ah)

    verts, faces, normals = surface_data
    print(f"Surface mesh: {len(verts)} verts, {len(faces)} triangles")
    _surf_dl = build_surface_dl(verts, faces, normals)

    cam = Camera()
    cam.target = prot_center.copy()
    VIEW_NAMES = ["Sticks", "Surface", "Spheres"]
    view_mode = 0
    hi_score = 0
    frame = 0
    paused = False
    menu_sel = 0
    pe, temp, contacts = 0.0, 300.0, 0

    state = ctx.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    lig_c = pos[ligand[0]]

    # runtime-adjustable parameters
    params = [
        {"name": "Restraint K",  "key": "rst_k",   "val": float(RESTRAINT_K),
         "min": 0, "max": 5000, "step": 50, "unit": "kJ/mol/nm2",
         "apply": lambda v: ctx.setParameter("rst_k", v)},
        {"name": "Force",        "key": "force",    "val": float(FORCE_MAG),
         "min": 50, "max": 2000, "step": 50, "unit": "kJ/mol/nm",
         "apply": None},
        {"name": "Friction",     "key": "friction", "val": float(FRICTION),
         "min": 0.5, "max": 20, "step": 0.5, "unit": "1/ps",
         "apply": lambda v: integrator.setFriction(v / unit.picosecond)},
        {"name": "Temperature",  "key": "temp",     "val": float(TEMPERATURE),
         "min": 10, "max": 1000, "step": 10, "unit": "K",
         "apply": lambda v: integrator.setTemperature(v * unit.kelvin)},
        {"name": "MD steps",     "key": "mdsteps",  "val": float(MD_STEPS),
         "min": 1, "max": 30, "step": 1, "unit": "/frame",
         "apply": None},
        {"name": "Water radius", "key": "wcut",     "val": float(WATER_VIS_CUT),
         "min": 0.5, "max": 5.0, "step": 0.25, "unit": "nm",
         "apply": None},
    ]

    while True:
        # ── Events ──
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); return
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    if paused:
                        paused = False
                        pygame.event.set_grab(True)
                        pygame.mouse.set_visible(False)
                    else:
                        pygame.quit(); return
                elif ev.key == K_p:
                    paused = not paused
                    pygame.event.set_grab(not paused)
                    pygame.mouse.set_visible(paused)
                elif paused:
                    if ev.key == K_UP:
                        menu_sel = (menu_sel - 1) % len(params)
                    elif ev.key == K_DOWN:
                        menu_sel = (menu_sel + 1) % len(params)
                    elif ev.key in (K_RIGHT, K_EQUALS, K_PLUS):
                        p = params[menu_sel]
                        p["val"] = min(p["max"], p["val"] + p["step"])
                        if p["apply"]:
                            p["apply"](p["val"])
                    elif ev.key in (K_LEFT, K_MINUS):
                        p = params[menu_sel]
                        p["val"] = max(p["min"], p["val"] - p["step"])
                        if p["apply"]:
                            p["apply"](p["val"])
                elif ev.key == K_v:
                    view_mode = (view_mode + 1) % len(VIEW_NAMES)
            elif ev.type == MOUSEWHEEL and not paused:
                cam.zoom(ev.y)

        if not paused:
            mdx, mdy = pygame.mouse.get_rel()
            cam.rotate(-mdx, mdy)

            cur_force = params[1]["val"]
            cur_mdsteps = int(params[4]["val"])
            cur_wcut = params[5]["val"]

            # ── Input → force (camera-relative) ──
            k = pygame.key.get_pressed()
            kw = bool(k[K_w]); ks = bool(k[K_s])
            ka = bool(k[K_a]); kd = bool(k[K_d])
            ksp = bool(k[K_SPACE]); ksh = bool(k[K_LSHIFT] or k[K_RSHIFT])
            active_keys = [kw, ka, ks, kd, ksp, ksh]

            f = np.zeros(3)
            if kw: f += cam.forward()
            if ks: f -= cam.forward()
            if kd: f += cam.right()
            if ka: f -= cam.right()
            if ksp: f[1] += 1
            if ksh: f[1] -= 1
            fn = np.linalg.norm(f)
            if fn > 0:
                f = f / fn * cur_force
            ctx.setParameter("fx", float(f[0]))
            ctx.setParameter("fy", float(f[1]))
            ctx.setParameter("fz", float(f[2]))

            integrator.step(cur_mdsteps)

            state = ctx.getState(getPositions=True, getEnergy=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            temp = 2 * ke / (3 * len(pos) * 8.314e-3)

            lig_c = pos[ligand[0]]
            cam.track(lig_c + np.array([0, 0.6, 0]))

            dists = np.linalg.norm(pos[prot_heavy] - lig_c, axis=1)
            contacts = int(np.sum(dists < CONTACT_CUT))
            hi_score = max(hi_score, contacts)
        else:
            active_keys = [False] * 6
            pygame.mouse.get_rel()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.apply()

        draw_grid(box_origin, box_lengths)
        draw_box(box_origin, box_lengths)
        if view_mode == 0:
            draw_protein_sticks(pos, bond_a, bond_b, bond_colors)
        elif view_mode == 1:
            draw_protein_surface()
        else:
            draw_protein_atoms(pos, prot_heavy, prot_elem)
        draw_water(pos, water_o, lig_c)
        draw_ligand(pos, ligand)
        draw_axes(aw, ah, cam)
        draw_crosshair(aw, ah)
        draw_hud(aw, ah, pdb_id, pdb_title, pe, temp, contacts,
                 hi_score, clock.get_fps(), active_keys, VIEW_NAMES[view_mode], frame)

        if paused:
            draw_pause_menu(aw, ah, params, menu_sel)

        pygame.display.flip()
        clock.tick(FPS_CAP)
        frame += 1


if __name__ == "__main__":
    main()
