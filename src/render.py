import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

from .config import CPK, CPK_TOON, VDW_RENDER, LIG_CPK, CL


_sdl = None
_sdl_toon = None
_quad = None
_surf_dl = None

PITCH_LIMIT = 85.0


def get_quad():
    return _quad


def set_surf_dl(dl):
    global _surf_dl
    _surf_dl = dl


def build_surface_dl(verts, faces, normals):
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    glColor3f(*CL["surface"])
    glBegin(GL_TRIANGLES)
    for f in faces:
        for i in f:
            glNormal3fv(normals[i])
            glVertex3fv(verts[i])
    glEnd()
    glEndList()
    return dl


class Camera:
    def __init__(self):
        self.yaw, self.pitch, self.dist = 0.0, 20.0, 6.0
        self.target = np.zeros(3)

    def rotate(self, dx, dy):
        self.yaw += dx
        self.pitch = np.clip(self.pitch + dy, -PITCH_LIMIT, PITCH_LIMIT)

    def zoom(self, d):
        self.dist = np.clip(self.dist - d, 2.0, 25.0)

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


def gl_init(aw, ah):
    global _sdl, _sdl_toon, _quad
    bg = CL["bg"]
    glClearColor(bg[0], bg[1], bg[2], 1.0)
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
    _sdl_toon = glGenLists(1)
    glNewList(_sdl_toon, GL_COMPILE); gluSphere(_quad, 1.0, 16, 8); glEndList()


def draw_protein_atoms(pos, prot_idx, prot_elem):
    by_elem = {}
    for idx, el in zip(prot_idx, prot_elem):
        if el == "H":
            continue
        by_elem.setdefault(el, []).append(idx)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
    for el, indices in by_elem.items():
        col = CPK.get(el, (0.35, 0.35, 0.35))
        r = VDW_RENDER.get(el, 0.07) * 1.15
        glColor3f(*col)
        for idx in indices:
            glPushMatrix()
            p = pos[idx]
            glTranslatef(float(p[0]), float(p[1]), float(p[2]))
            glScalef(r, r, r)
            glCallList(_sdl)
            glPopMatrix()
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)


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


def draw_protein_backbone(pos, ca_traces):
    glDisable(GL_LIGHTING)
    glLineWidth(8.0)
    glColor3f(*CL["backbone_dim"])
    for trace in ca_traces:
        if len(trace) < 2:
            continue
        glBegin(GL_LINE_STRIP)
        for idx in trace:
            p = pos[idx]
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()
    glLineWidth(4.0)
    glColor3f(*CL["backbone"])
    for trace in ca_traces:
        if len(trace) < 2:
            continue
        glBegin(GL_LINE_STRIP)
        for idx in trace:
            p = pos[idx]
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def draw_protein_surface():
    if _surf_dl is not None:
        glCallList(_surf_dl)


def draw_water(pos, water_o, lig_pos, wcut):
    wp_all = pos[water_o]
    dists = np.linalg.norm(wp_all - lig_pos, axis=1)
    nearby = wp_all[dists < wcut].astype(np.float32)
    if len(nearby) == 0:
        return
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(*CL["water"])
    glPointSize(2.0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, nearby)
    glDrawArrays(GL_POINTS, 0, len(nearby))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisable(GL_BLEND); glEnable(GL_LIGHTING)


def draw_ligand(pos, lig, lig_elem=None, lig_bond_a=None, lig_bond_b=None,
                lig_bond_colors=None, is_ion=False, active_set=None,
                ligand_style=0, show_glow=True):
    if is_ion:
        for idx in lig:
            p = pos[idx]
            glPushMatrix()
            glTranslatef(float(p[0]), float(p[1]), float(p[2]))
            glColor3f(*CL["ion_solid"])
            gluSphere(_quad, 0.18, 16, 8)
            glPopMatrix()
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for idx in lig:
            if active_set is not None and idx not in active_set:
                continue
            p = pos[idx]
            glColor4f(*CL["ion_glow"])
            glPushMatrix()
            glTranslatef(float(p[0]), float(p[1]), float(p[2]))
            gluSphere(_quad, 0.25, 12, 6)
            glPopMatrix()
        glDisable(GL_BLEND); glEnable(GL_LIGHTING)
    elif len(lig) == 1:
        p = pos[lig[0]]
        glPushMatrix()
        glTranslatef(float(p[0]), float(p[1]), float(p[2]))
        glColor3f(*CL["probe_solid"])
        gluSphere(_quad, 0.18, 16, 8)
        glPopMatrix()
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(*CL["probe_glow"])
        glPushMatrix()
        glTranslatef(float(p[0]), float(p[1]), float(p[2]))
        gluSphere(_quad, 0.25, 12, 6)
        glPopMatrix()
        glDisable(GL_BLEND); glEnable(GL_LIGHTING)
    else:
        if ligand_style == 1:
            glDisable(GL_LIGHTING)
            if lig_bond_a is not None and len(lig_bond_a) > 0:
                draw_protein_sticks(pos, lig_bond_a, lig_bond_b, lig_bond_colors)
                glDisable(GL_LIGHTING)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            for idx, el in zip(lig, lig_elem):
                glColor3f(*LIG_CPK.get(el, LIG_CPK["C"]))
                p = pos[idx]
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()
            glEnable(GL_LIGHTING)
            return

        if lig_bond_a is not None and len(lig_bond_a) > 0:
            draw_protein_sticks(pos, lig_bond_a, lig_bond_b, lig_bond_colors)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
        for idx, el in zip(lig, lig_elem):
            col = LIG_CPK.get(el, LIG_CPK["C"])
            scale = 0.65 if el == "H" else 1.45
            r = VDW_RENDER.get(el, 0.07) * scale
            glColor3f(*col)
            glPushMatrix()
            p = pos[idx]
            glTranslatef(float(p[0]), float(p[1]), float(p[2]))
            glScalef(r, r, r)
            glCallList(_sdl_toon)
            glPopMatrix()
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)
        if show_glow:
            center = pos[lig].mean(axis=0)
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(*CL["lig_glow"])
            glPushMatrix()
            glTranslatef(float(center[0]), float(center[1]), float(center[2]))
            gluSphere(_quad, 0.5, 16, 8)
            glPopMatrix()
            glDisable(GL_BLEND); glEnable(GL_LIGHTING)


def draw_grid(origin, lengths, step=0.5):
    y = origin[1]
    x0, z0 = origin[0], origin[2]
    x1, z1 = x0 + lengths[0], z0 + lengths[2]
    glDisable(GL_LIGHTING)
    glColor3f(*CL["grid"])
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
    glColor3f(*CL["box"])
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
    for (dx, dy, dz), key in [((1,0,0), "axis_x"),
                               ((0,1,0), "axis_y"),
                               ((0,0,1), "axis_z")]:
        glColor3f(*CL[key]); glBegin(GL_LINES)
        glVertex3f(0,0,0); glVertex3f(dx,dy,dz); glEnd()
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()
    glViewport(0, 0, aw, ah)


def draw_crosshair(aw, ah, size=12, color=None):
    gap = max(2, size // 3)
    cx, cy = aw // 2, ah // 2
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, aw, 0, ah, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    if color is not None:
        glColor4f(*color)
    else:
        glColor4f(*CL["crosshair"])
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


def draw_scanlines(aw, ah):
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, aw, 0, ah, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(*CL["scanline"])
    glBegin(GL_LINES)
    for y in range(0, ah, 3):
        glVertex2i(0, y); glVertex2i(aw, y)
    glEnd()
    glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glPopMatrix()


def draw_selected_residue(pos, current_target):
    center = pos[current_target].mean(axis=0)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(*CL["sel_glow"])
    glPushMatrix()
    glTranslatef(float(center[0]), float(center[1]), float(center[2]))
    gluSphere(_quad, 0.5, 16, 8)
    glPopMatrix()
    for i in current_target:
        p = pos[i]
        glColor4f(*CL["sel_atom"])
        glPushMatrix()
        glTranslatef(float(p[0]), float(p[1]), float(p[2]))
        gluSphere(_quad, 0.08, 8, 4)
        glPopMatrix()
    glDisable(GL_BLEND); glEnable(GL_LIGHTING)
