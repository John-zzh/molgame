#!/usr/bin/env python3
"""
MolGame — Pixel Edition
Real protein MD + retro pixel-art HUD.  WASD steer, mouse look, scroll zoom.
Usage:  python start.py [--pdb 1UBQ]
        python start.py [--pdb 4HJO --ligand AQ4]
        python start.py [--pdb-file protein.pdb --ligand-file ligand.sdf]
"""

import sys, math, os, argparse, time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from openmm import unit

from src.config import (W, H, FPS_CAP, CONTACT_CUT, CPK, LIG_CPK,
                         CROSS_COLORS, CROSS_COLOR_NAMES,
                         load_config, save_config)
from src.prepare import prepare, download_pdb, read_pdb_title
from src.render import (Camera, gl_init, build_surface_dl, set_surf_dl,
                         draw_protein_atoms, draw_protein_sticks,
                         draw_protein_backbone, draw_protein_surface,
                         draw_water, draw_ligand,
                         draw_grid, draw_box, draw_axes, draw_crosshair,
                         draw_scanlines, draw_selected_residue)
from src.hud import draw_hud, draw_pause_menu


def main():
    parser = argparse.ArgumentParser(description="MolGame — Molecular Dynamics Game")
    parser.add_argument("--pdb", default="1UBQ", help="PDB ID (default: 1UBQ)")
    parser.add_argument("--pdb-file", default=None,
                        help="Local protein PDB file instead of downloading by PDB ID")
    parser.add_argument("--ligand", default=None,
                        help="Ligand residue name in PDB (e.g. ZMA, ATP)")
    parser.add_argument("--ligand-file", default=None,
                        help="Local ligand .sdf or .mol2 file with explicit H and 3D coordinates")
    parser.add_argument("--ion", default=None,
                        help="Control an existing ion in PDB (e.g. CA, NA, MG)")
    args = parser.parse_args()
    lig_name = args.ligand.upper() if args.ligand else None
    ligand_file = os.path.abspath(args.ligand_file) if args.ligand_file else None
    ion_name = args.ion.upper() if args.ion else None

    if lig_name and ligand_file:
        parser.error("--ligand and --ligand-file are mutually exclusive")
    if ion_name and (lig_name or ligand_file):
        parser.error("--ion cannot be combined with --ligand or --ligand-file")
    if ligand_file and not os.path.exists(ligand_file):
        parser.error(f"--ligand-file not found: {ligand_file}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pdb_file:
        pdb_file = os.path.abspath(args.pdb_file)
        if not os.path.exists(pdb_file):
            parser.error(f"--pdb-file not found: {pdb_file}")
        pdb_id = os.path.splitext(os.path.basename(pdb_file))[0].upper()
    else:
        pdb_id = args.pdb.upper()
        pdb_file = os.path.join(base_dir, f"{pdb_id.lower()}.pdb")
        if not os.path.exists(pdb_file):
            download_pdb(pdb_id, pdb_file)

    pdb_title = read_pdb_title(pdb_file)
    print(f"Protein: {pdb_id}  {pdb_title}")

    cfg = load_config()

    (ctx, integrator, prot_idx, prot_elem,
     water_o, ligand, prot_center, surface_data,
     box_origin, box_lengths, prot_bonds,
     lig_elem, lig_bonds,
     steer_force, steer_map, torque_force, torque_map,
     res_atoms, chain_atoms, ca_traces, atom_records) = prepare(
        pdb_file, cfg, lig_name, ion_name, ligand_file)
    has_real_ligand = lig_name is not None or ligand_file is not None

    if ion_name and len(ligand) > 1:
        current_target = np.array([ligand[0]])
    else:
        current_target = ligand.copy()
    home_target = current_target.copy()
    control_mode = "ligand"  # "ligand" → "free" → "residue" → "ligand"
    atom_to_res = {}
    atom_to_chain = {}
    for rkey, indices in res_atoms.items():
        for idx in indices:
            atom_to_res[idx] = rkey
    for ckey, indices in chain_atoms.items():
        for idx in indices:
            atom_to_chain[idx] = ckey

    def switch_target(new_target):
        nonlocal current_target
        for idx in current_target:
            if idx in steer_map:
                steer_force.setParticleParameters(steer_map[idx], idx, [0.0])
        for idx in new_target:
            if idx in steer_map:
                steer_force.setParticleParameters(steer_map[idx], idx, [1.0])
        steer_force.updateParametersInContext(ctx)
        current_target = np.array(new_target)

    torque_atoms = []

    def clear_torque():
        nonlocal torque_atoms
        if not torque_atoms:
            return
        for idx in torque_atoms:
            if idx in torque_map:
                torque_force.setParticleParameters(torque_map[idx], idx, [0.0, 0.0, 0.0])
        torque_force.updateParametersInContext(ctx)
        torque_atoms = []

    def apply_target_torque(target, positions, key_state, hat_x=0, hat_y=0):
        nonlocal torque_atoms
        clear_torque()
        if len(target) <= 1:
            return
        axes = []
        if key_state[K_UP] or hat_y > 0:
            axes.append(cam.right())
        if key_state[K_DOWN] or hat_y < 0:
            axes.append(-cam.right())
        if key_state[K_LEFT] or hat_x < 0:
            axes.append(np.array([0.0, 1.0, 0.0]))
        if key_state[K_RIGHT] or hat_x > 0:
            axes.append(np.array([0.0, -1.0, 0.0]))
        if not axes:
            return
        com = positions[target].mean(axis=0)
        tangents = []
        atom_ids = []
        for idx in target:
            r = positions[idx] - com
            t = np.zeros(3)
            for axis in axes:
                t += np.cross(axis, r)
            tn = np.linalg.norm(t)
            if tn > 1e-6 and idx in torque_map:
                tangents.append(t)
                atom_ids.append(int(idx))
        if not tangents:
            return
        avg = np.mean([np.linalg.norm(t) for t in tangents])
        scale = cfg["torque_force"] / max(avg, 1e-6)
        for idx, t in zip(atom_ids, tangents):
            vec = (t * scale).tolist()
            torque_force.setParticleParameters(torque_map[idx], idx, vec)
        torque_force.updateParametersInContext(ctx)
        torque_atoms = atom_ids

    elem_map = {int(i): str(e) for i, e in zip(prot_idx, prot_elem)}
    if has_real_ligand:
        for i, e in zip(ligand, lig_elem):
            elem_map[int(i)] = str(e)
    bond_a = np.array([a for a, b in prot_bonds], dtype=np.int32)
    bond_b = np.array([b for a, b in prot_bonds], dtype=np.int32)
    bond_colors = np.empty((len(prot_bonds) * 2, 3), dtype=np.float32)
    for i, (a, b) in enumerate(prot_bonds):
        bond_colors[i * 2] = CPK.get(elem_map[a], CPK["C"])
        bond_colors[i * 2 + 1] = CPK.get(elem_map[b], CPK["C"])

    if has_real_ligand and len(lig_bonds) > 0:
        lig_bond_a = np.array([a for a, b in lig_bonds], dtype=np.int32)
        lig_bond_b = np.array([b for a, b in lig_bonds], dtype=np.int32)
        lig_bond_colors = np.empty((len(lig_bonds) * 2, 3), dtype=np.float32)
        for i, (a, b) in enumerate(lig_bonds):
            lig_bond_colors[i * 2] = LIG_CPK.get(elem_map.get(a, "C"), LIG_CPK["C"])
            lig_bond_colors[i * 2 + 1] = LIG_CPK.get(elem_map.get(b, "C"), LIG_CPK["C"])
    else:
        lig_bond_a, lig_bond_b, lig_bond_colors = None, None, None

    # ── Pygame + OpenGL ──
    pygame.init(); pygame.font.init()
    fullscreen = False
    screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"MolGame — {pdb_id}")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    if hasattr(pygame.mouse, "set_relative_mode"):
        pygame.mouse.set_relative_mode(True)
    clock = pygame.time.Clock()

    vp = glGetIntegerv(GL_VIEWPORT)
    aw, ah = int(vp[2]), int(vp[3])
    gl_init(aw, ah)

    verts, faces, normals = surface_data
    print(f"Surface mesh: {len(verts)} verts, {len(faces)} triangles")
    set_surf_dl(build_surface_dl(verts, faces, normals))

    # ── Gamepad ──
    pygame.joystick.init()
    pad = None
    if pygame.joystick.get_count() > 0:
        pad = pygame.joystick.Joystick(0)
        pad.init()
        print(f"Gamepad: {pad.get_name()}  axes={pad.get_numaxes()} buttons={pad.get_numbuttons()}")

    def pad_axis(idx):
        if pad is None or idx >= pad.get_numaxes():
            return 0.0
        v = pad.get_axis(idx)
        return v if abs(v) > cfg["pad_deadzone"] else 0.0

    def set_mouse_capture(enabled):
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        if hasattr(pygame.mouse, "set_relative_mode"):
            pygame.mouse.set_relative_mode(enabled)
        pygame.mouse.get_rel()

    def toggle_fullscreen():
        nonlocal fullscreen, aw, ah
        fullscreen = not fullscreen
        if fullscreen:
            screen = pygame.display.set_mode((0, 0), DOUBLEBUF | OPENGL | FULLSCREEN)
        else:
            screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
        vp = glGetIntegerv(GL_VIEWPORT)
        aw, ah = int(vp[2]), int(vp[3])
        gl_init(aw, ah)
        set_surf_dl(build_surface_dl(*surface_data))
        set_mouse_capture(True)

    cam = Camera()
    cam.target = prot_center.copy()
    VIEW_NAMES = ["Sticks", "Surface", "Backbone"]
    LIGAND_STYLE_NAMES = ["Ball", "Line"]
    SELECT_SCOPE_NAMES = ["Residue", "Chain"]
    FORCE_MODE_NAMES = ["Total", "Per atom"]
    SAVE_WATER_NAMES = ["No", "Yes"]
    view_mode = 0
    hi_score = 0
    frame = 0
    paused = False
    menu_sel = 0
    pe, pe_no_steer, temp, contacts = 0.0, 0.0, 300.0, 0
    nan_frames = 0
    nan_total = 0
    nan_pause_reported = False
    prev_x_pressed = False
    prev_a_pressed = False

    state = ctx.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    prev_pos = pos.copy()
    lig_c = pos[ligand].mean(axis=0) if len(ligand) > 1 else pos[ligand[0]]
    cur_wcut = cfg["water_radius"]

    params = [
        {"name": "Restraint K",   "cfg": "restraint_k", "val": cfg["restraint_k"],
         "min": 0, "max": 5000, "step": 50, "unit": "kJ/mol/nm2",
         "apply": lambda v: ctx.setParameter("rst_k", v)},
        {"name": "Force",         "cfg": "force",       "val": cfg["force"],
         "min": 50, "max": 50000 if has_real_ligand else 2000,
         "step": 10, "unit": "kJ/mol/nm",
         "apply": None},
        {"name": "Force mode",    "cfg": "force_mode",  "val": cfg["force_mode"],
         "min": 0, "max": len(FORCE_MODE_NAMES) - 1, "step": 1, "unit": "",
         "choices": FORCE_MODE_NAMES,
         "apply": None},
        {"name": "Lig force x",   "cfg": "ligand_force_scale", "val": cfg["ligand_force_scale"],
         "min": 1.0, "max": 50.0, "step": 1.0, "unit": "x",
         "apply": None},
        {"name": "Torque",        "cfg": "torque_force", "val": cfg["torque_force"],
         "min": 0.0, "max": 5000.0, "step": 10.0, "unit": "kJ/mol/nm",
         "apply": None},
        {"name": "Friction",      "cfg": "friction",    "val": cfg["friction"],
         "min": 0.5, "max": 20, "step": 0.5, "unit": "1/ps",
         "apply": lambda v: integrator.setFriction(v / unit.picosecond)},
        {"name": "Temperature",   "cfg": "temperature", "val": cfg["temperature"],
         "min": 10, "max": 1000, "step": 10, "unit": "K",
         "apply": lambda v: integrator.setTemperature(v * unit.kelvin)},
        {"name": "MD steps",      "cfg": "md_steps",    "val": cfg["md_steps"],
         "min": 1, "max": 30, "step": 1, "unit": "/frame",
         "apply": None},
        {"name": "Timestep",      "cfg": "timestep_fs", "val": cfg["timestep_fs"],
         "min": 0.5, "max": 4.0, "step": 0.5, "unit": "fs",
         "apply": lambda v: integrator.setStepSize(v * unit.femtoseconds)},
        {"name": "Water radius",  "cfg": "water_radius","val": cfg["water_radius"],
         "min": 0.5, "max": 5.0, "step": 0.25, "unit": "nm",
         "apply": None},
        {"name": "Mouse sens",    "cfg": "mouse_sens",  "val": cfg["mouse_sens"],
         "min": 0.05, "max": 1.0, "step": 0.05, "unit": "",
         "apply": None},
        {"name": "Scroll sens",   "cfg": "scroll_sens", "val": cfg["scroll_sens"],
         "min": 0.1, "max": 2.0, "step": 0.1, "unit": "",
         "apply": None},
        {"name": "Pad look sens", "cfg": "pad_look_sens","val": cfg["pad_look_sens"],
         "min": 0.5, "max": 10.0, "step": 0.5, "unit": "",
         "apply": None},
        {"name": "Pad zoom sens", "cfg": "pad_zoom_sens","val": cfg["pad_zoom_sens"],
         "min": 0.05, "max": 1.0, "step": 0.05, "unit": "",
         "apply": None},
        {"name": "Aim offset Y", "cfg": "aim_offset_y", "val": cfg["aim_offset_y"],
         "min": 0.0, "max": 3.0, "step": 0.1, "unit": "nm",
         "apply": None},
        {"name": "Ligand style", "cfg": "ligand_style", "val": cfg["ligand_style"],
         "min": 0, "max": len(LIGAND_STYLE_NAMES) - 1, "step": 1, "unit": "",
         "choices": LIGAND_STYLE_NAMES,
         "apply": None},
        {"name": "Select scope", "cfg": "select_scope", "val": cfg["select_scope"],
         "min": 0, "max": len(SELECT_SCOPE_NAMES) - 1, "step": 1, "unit": "",
         "choices": SELECT_SCOPE_NAMES,
         "apply": None},
        {"name": "Save water", "cfg": "save_water", "val": cfg["save_water"],
         "min": 0, "max": len(SAVE_WATER_NAMES) - 1, "step": 1, "unit": "",
         "choices": SAVE_WATER_NAMES,
         "apply": None},
        {"name": "Cross size",  "cfg": "cross_size",   "val": cfg["cross_size"],
         "min": 4, "max": 40, "step": 2, "unit": "px",
         "apply": None},
        {"name": "Cross color", "cfg": "cross_color",  "val": cfg["cross_color"],
         "min": 0, "max": len(CROSS_COLORS) - 1, "step": 1, "unit": "",
         "choices": CROSS_COLOR_NAMES,
         "apply": None},
    ]

    def sync_cfg():
        for p in params:
            cfg[p["cfg"]] = p["val"]
        save_config(cfg)

    for p in params:
        cfg[p["cfg"]] = p["val"]

    def pdb_atom_name(name, element):
        n = str(name)[:4]
        e = str(element).strip()
        if len(n) < 4 and len(e) < 2:
            return f" {n:<3s}"
        return f"{n:<4s}"

    def pdb_resid(value, fallback):
        try:
            return max(-999, min(9999, int(str(value).strip())))
        except Exception:
            return ((fallback - 1) % 9999) + 1

    def save_current_structure(cur_pos, include_water=False):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "all" if include_water else "dry"
        path = os.path.abspath(f"molgame_snapshot_{stamp}_{suffix}.pdb")
        serial = 1
        with open(path, "w") as f:
            f.write("REMARK Saved by MolGame\n")
            f.write(f"REMARK Water included: {'yes' if include_water else 'no'}\n")
            for rec in atom_records:
                if rec["is_water"] and not include_water:
                    continue
                idx = rec["index"]
                if idx >= len(cur_pos):
                    continue
                p = cur_pos[idx] * 10.0
                record = "ATOM" if rec["is_protein"] else "HETATM"
                name = pdb_atom_name(rec["name"], rec["element"])
                resname = str(rec["resname"])[:3].rjust(3)
                chain = str(rec["chain"] or "A")[:1]
                resid = pdb_resid(rec["resid"], serial)
                elem = str(rec["element"]).strip()[:2].rjust(2)
                f.write(
                    f"{record:<6}{serial:5d} {name} {resname} {chain}{resid:4d}"
                    f"    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                    f"  1.00  0.00          {elem}\n"
                )
                serial += 1
            f.write("END\n")
        print(f"Saved structure: {path}")

    while True:
        # ── Events ──
        do_select = False
        mouse_dx, mouse_dy = 0, 0
        for ev in pygame.event.get():
            if ev.type == QUIT:
                sync_cfg(); pygame.quit(); return
            elif ev.type == MOUSEMOTION and not paused:
                mouse_dx += ev.rel[0]
                mouse_dy += ev.rel[1]
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    if paused:
                        paused = False
                        sync_cfg()
                        set_mouse_capture(True)
                    else:
                        sync_cfg(); pygame.quit(); return
                elif ev.key == K_p:
                    paused = not paused
                    if not paused:
                        sync_cfg()
                    set_mouse_capture(not paused)
                elif paused:
                    if ev.key == K_s:
                        save_current_structure(pos, include_water=bool(int(cfg["save_water"])))
                    elif ev.key == K_UP:
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
                elif ev.key == K_l and has_real_ligand:
                    cfg["ligand_style"] = (int(cfg["ligand_style"]) + 1) % len(LIGAND_STYLE_NAMES)
                    for p in params:
                        if p["cfg"] == "ligand_style":
                            p["val"] = cfg["ligand_style"]
                            break
                elif ev.key == K_F11:
                    toggle_fullscreen()
                elif ev.key == K_x:
                    do_select = True  # fallback if IME doesn't eat it
            elif ev.type == MOUSEWHEEL and not paused:
                cam.zoom(ev.y * cfg["scroll_sens"])
            elif ev.type == JOYBUTTONDOWN:
                if ev.button == 0:  # A
                    paused = not paused
                    if not paused:
                        sync_cfg()
                    set_mouse_capture(not paused)
                elif ev.button == 1 and not paused:  # B
                    view_mode = (view_mode + 1) % len(VIEW_NAMES)
                elif ev.button == 2 and not paused:  # X
                    do_select = True
                elif ev.button == 3 and not paused:  # Y
                    toggle_fullscreen()
                elif ev.button == 6:  # Start/Menu
                    if paused:
                        paused = False
                        sync_cfg()
                        set_mouse_capture(True)
                    else:
                        sync_cfg(); pygame.quit(); return
                elif paused:
                    if ev.button == 9:  # LB
                        p = params[menu_sel]
                        p["val"] = max(p["min"], p["val"] - p["step"])
                        if p["apply"]: p["apply"](p["val"])
                    elif ev.button == 10:  # RB
                        p = params[menu_sel]
                        p["val"] = min(p["max"], p["val"] + p["step"])
                        if p["apply"]: p["apply"](p["val"])
            elif ev.type == JOYHATMOTION and paused:
                hx, hy = ev.value
                if hy == 1:
                    menu_sel = (menu_sel - 1) % len(params)
                elif hy == -1:
                    menu_sel = (menu_sel + 1) % len(params)
                elif hx == 1:
                    p = params[menu_sel]
                    p["val"] = min(p["max"], p["val"] + p["step"])
                    if p["apply"]: p["apply"](p["val"])
                elif hx == -1:
                    p = params[menu_sel]
                    p["val"] = max(p["min"], p["val"] - p["step"])
                    if p["apply"]: p["apply"](p["val"])
            elif ev.type == JOYDEVICEADDED:
                if pad is None:
                    pad = pygame.joystick.Joystick(ev.device_index)
                    pad.init()
                    print(f"Gamepad connected: {pad.get_name()}")
            elif ev.type == JOYDEVICEREMOVED:
                pad = None
                print("Gamepad disconnected")

        if pad and pad.get_numbuttons() > 0:
            a_now = pad.get_button(0)
            if a_now and not prev_a_pressed:
                paused = not paused
                if not paused:
                    sync_cfg()
                set_mouse_capture(not paused)
            prev_a_pressed = a_now

        if not paused:
            if (not pygame.event.get_grab() or
                    (hasattr(pygame.mouse, "get_relative_mode")
                     and not pygame.mouse.get_relative_mode())):
                set_mouse_capture(True)
            mdx, mdy = mouse_dx, mouse_dy
            ms = cfg["mouse_sens"]
            cam.rotate(-mdx * ms, mdy * ms)

            pls = cfg["pad_look_sens"]
            cam.rotate(-pad_axis(2) * pls, pad_axis(3) * pls)
            pzs = cfg["pad_zoom_sens"]
            if pad and pad.get_numbuttons() > 10:
                if pad.get_button(9): cam.zoom(pzs)
                if pad.get_button(10): cam.zoom(-pzs)

            cur_force = cfg["force"]
            cur_mdsteps = int(cfg["md_steps"])
            cur_wcut = cfg["water_radius"]

            k = pygame.key.get_pressed()
            kw = bool(k[K_w]); ks = bool(k[K_s])
            ka = bool(k[K_a]); kd = bool(k[K_d])
            ksp = bool(k[K_SPACE]); ksh = bool(k[K_LSHIFT] or k[K_RSHIFT])
            active_keys = [kw, ka, ks, kd, ksp, ksh]

            x_now = bool(k[K_x])
            if x_now and not prev_x_pressed:
                do_select = True
            prev_x_pressed = x_now

            lx, ly = pad_axis(0), pad_axis(1)
            lt = (pad_axis(4) + 1) * 0.5
            rt = (pad_axis(5) + 1) * 0.5
            hx, hy = pad.get_hat(0) if pad and pad.get_numhats() > 0 else (0, 0)

            if control_mode == "free":
                clear_torque()
                cam_speed = 0.08
                mv = np.zeros(3)
                if kw: mv += cam.forward()
                if ks: mv -= cam.forward()
                if kd: mv += cam.right()
                if ka: mv -= cam.right()
                if ksp: mv[1] += 1
                if ksh: mv[1] -= 1
                mv -= cam.forward() * ly
                mv += cam.right() * lx
                mv[1] += rt - lt
                mvn = np.linalg.norm(mv)
                if mvn > 0:
                    cam.target += mv / mvn * cam_speed
            else:
                f = np.zeros(3)
                if kw: f += cam.forward()
                if ks: f -= cam.forward()
                if kd: f += cam.right()
                if ka: f -= cam.right()
                if ksp: f[1] += 1
                if ksh: f[1] -= 1
                f -= cam.forward() * ly
                f += cam.right() * lx
                f[1] += rt - lt
                fn = np.linalg.norm(f)
                if fn > 0:
                    f = f / fn * cur_force
                    if has_real_ligand and np.array_equal(current_target, home_target):
                        f *= cfg["ligand_force_scale"]
                    if len(current_target) > 1 and int(cfg["force_mode"]) == 0:
                        f /= len(current_target)
                ctx.setParameter("fx", float(f[0]))
                ctx.setParameter("fy", float(f[1]))
                ctx.setParameter("fz", float(f[2]))
                apply_target_torque(current_target, pos, k, hx, hy)

            try:
                integrator.step(cur_mdsteps)
                state = ctx.getState(getPositions=True, getEnergy=True)
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                if np.any(np.isnan(pos)):
                    raise RuntimeError("NaN")
                prev_pos = pos.copy()
                nan_frames = 0
                nan_pause_reported = False
                pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                steer_state = ctx.getState(getEnergy=True, groups={31})
                steer_pe = steer_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                pe_no_steer = pe - steer_pe
                ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
                temp = 2 * ke / (3 * len(pos) * 8.314e-3)
            except Exception:
                try:
                    ctx.reinitialize(preserveState=False)
                except Exception:
                    pass
                ctx.setParameter("fx", 0.0)
                ctx.setParameter("fy", 0.0)
                ctx.setParameter("fz", 0.0)
                clear_torque()
                ctx.setPositions(prev_pos * unit.nanometers)
                ctx.setVelocitiesToTemperature(1 * unit.kelvin)
                pos = prev_pos
                nan_frames += 1
                nan_total += 1
                if nan_total <= 5 or nan_total % 50 == 0:
                    print(f"[NaN recovery #{nan_total}] positions restored")
                if nan_frames >= 10 and not nan_pause_reported:
                    paused = True
                    nan_pause_reported = True
                    cfg["md_steps"] = 1
                    for p in params:
                        if p["cfg"] == "md_steps":
                            p["val"] = 1
                            break
                    set_mouse_capture(False)
                    print("[NaN recovery] paused after 10 consecutive failed frames; MD steps set to 1")

            if do_select:
                if control_mode == "ligand":
                    control_mode = "free"
                    ctx.setParameter("fx", 0.0)
                    ctx.setParameter("fy", 0.0)
                    ctx.setParameter("fz", 0.0)
                elif control_mode == "free":
                    yr_r = math.radians(cam.yaw)
                    pr_r = math.radians(cam.pitch)
                    eye = cam.target + cam.dist * np.array([
                        math.sin(yr_r)*math.cos(pr_r), math.sin(pr_r),
                        math.cos(yr_r)*math.cos(pr_r)])
                    ray_dir = cam.target - eye
                    ray_dir /= np.linalg.norm(ray_dir)

                    hit_type = None
                    hit_data = None

                    if ion_name and len(ligand) > 1:
                        ion_pos = pos[ligand]
                        ion_off = ion_pos - eye
                        ion_along = np.dot(ion_off, ray_dir)
                        ion_mask = ion_along > 0
                        if ion_mask.any():
                            perp = ion_off[ion_mask] - ion_along[ion_mask, None] * ray_dir
                            dr = np.linalg.norm(perp, axis=1)
                            bi = np.argmin(dr)
                            if dr[bi] < 0.4:
                                hit_type = "ion"
                                hit_data = int(ligand[np.where(ion_mask)[0][bi]])

                    if hit_type is None:
                        atom_pos = pos[prot_idx]
                        offsets = atom_pos - eye
                        along = np.dot(offsets, ray_dir)
                        mask = along > 0
                        if mask.any():
                            perp = offsets[mask] - along[mask, None] * ray_dir
                            dr = np.linalg.norm(perp, axis=1)
                            bi = np.argmin(dr)
                            hit_type = "residue"
                            nearest_atom = int(prot_idx[np.where(mask)[0][bi]])
                            hit_data = nearest_atom

                    if hit_type == "ion":
                        switch_target([hit_data])
                        home_target = current_target.copy()
                        control_mode = "ligand"
                    elif hit_type == "residue" and hit_data in atom_to_res:
                        if int(cfg["select_scope"]) == 1 and hit_data in atom_to_chain:
                            ckey = atom_to_chain[hit_data]
                            switch_target(chain_atoms[ckey])
                        else:
                            rkey = atom_to_res[hit_data]
                            switch_target(res_atoms[rkey])
                        control_mode = "residue"
                elif control_mode == "residue":
                    switch_target(home_target)
                    control_mode = "ligand"

            lig_c = pos[ligand].mean(axis=0) if len(ligand) > 1 else pos[ligand[0]]
            if control_mode != "free":
                track_c = pos[current_target].mean(axis=0) if len(current_target) > 1 else pos[current_target[0]]
                cam.track(track_c + np.array([0, cfg["aim_offset_y"], 0]))

            dists = np.linalg.norm(pos[prot_idx] - lig_c, axis=1)
            contacts = int(np.sum(dists < CONTACT_CUT))
            hi_score = max(hi_score, contacts)
        else:
            active_keys = [False] * 6
            clear_torque()
            pygame.mouse.get_rel()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.apply()

        draw_grid(box_origin, box_lengths)
        draw_box(box_origin, box_lengths)
        if view_mode == 0:
            draw_protein_sticks(pos, bond_a, bond_b, bond_colors)
        elif view_mode == 1:
            draw_protein_surface()
        elif view_mode == 2:
            draw_protein_backbone(pos, ca_traces)
        if not has_real_ligand:
            draw_water(pos, water_o, lig_c, cur_wcut)
        draw_ligand(pos, ligand, lig_elem, lig_bond_a, lig_bond_b, lig_bond_colors,
                    is_ion=ion_name is not None,
                    active_set=set(current_target) if ion_name else None,
                    ligand_style=int(cfg["ligand_style"]),
                    show_glow=control_mode != "free")
        if control_mode == "residue":
            draw_selected_residue(pos, current_target)
        draw_axes(aw, ah, cam)
        cross_color = CROSS_COLORS[2] if control_mode == "free" else CROSS_COLORS[int(cfg["cross_color"])]
        draw_crosshair(aw, ah, size=int(cfg["cross_size"]), color=cross_color)
        draw_hud(aw, ah, pdb_id, pdb_title, pe, temp, contacts,
                 hi_score, clock.get_fps(), active_keys, VIEW_NAMES[view_mode],
                 pe_no_steer,
                 SELECT_SCOPE_NAMES[int(cfg["select_scope"])],
                 FORCE_MODE_NAMES[int(cfg["force_mode"])], frame)

        if paused:
            draw_pause_menu(aw, ah, params, menu_sel)

        pygame.display.flip()
        clock.tick(FPS_CAP)
        frame += 1


if __name__ == "__main__":
    main()
