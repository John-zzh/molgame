import pygame
from OpenGL.GL import *

from .config import PX


_hud_font = None
_hud_font_sm = None


def _txt(text, color, big=True):
    global _hud_font, _hud_font_sm
    if _hud_font is None:
        _hud_font = pygame.font.SysFont("menlo", 19, bold=True)
        _hud_font_sm = pygame.font.SysFont("menlo", 15, bold=True)
    font = _hud_font if big else _hud_font_sm
    return font.render(text, False, color)


def draw_hud(aw, ah, pdb_id, pdb_title, pe, temp, contacts,
             hi_score, fps, active_keys, view_name, pe_no_steer, select_scope,
             force_mode, frame):
    pad = 10
    lh = 25
    bw = 420

    if contacts >= 12:
        flash = (frame // 8) % 2
        st_txt, st_col = "IN POCKET!", PX["ok"] if flash else PX["title"]
    elif contacts >= 4:
        st_txt, st_col = "CONTACT!", PX["warn"]
    else:
        st_txt, st_col = "EXPLORING", PX["dim"]

    bh = 400
    surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
    surf.fill((6, 6, 16, 210))
    pygame.draw.rect(surf, PX["border2"], (0, 0, bw, bh), 1)

    y = pad

    surf.blit(_txt(f"MOLGAME  {pdb_id}", PX["title"]), (pad, y))
    y += lh

    bar_x = pad
    seg_w, seg_h = 14, 14
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

    surf.blit(_txt(st_txt, st_col), (pad, y))
    hi_t = _txt(f"HI {hi_score}", PX["ok"], big=False)
    surf.blit(hi_t, (bw - pad - hi_t.get_width(), y + 2))
    y += lh + 2

    pygame.draw.line(surf, PX["sep"], (pad, y), (bw - pad, y))
    y += 5

    surf.blit(_txt(f"PE total {pe:10.0f} kJ/mol", PX["dim"], big=False), (pad, y))
    y += lh - 4
    surf.blit(_txt(f"PE clean {pe_no_steer:10.0f} kJ/mol", PX["dim"], big=False), (pad, y))
    y += lh - 4
    surf.blit(_txt(f"Temp {temp:5.0f}K  FPS {fps:3.0f}", PX["dim"], big=False), (pad, y))
    y += lh - 4
    vn = view_name
    surf.blit(_txt(f"View: {vn} [V]", PX["dim"], big=False), (pad, y))
    y += lh - 2
    surf.blit(_txt(f"Select: {select_scope}", PX["dim"], big=False), (pad, y))
    y += lh - 2
    surf.blit(_txt(f"Force: {force_mode}", PX["dim"], big=False), (pad, y))
    y += lh - 2

    pygame.draw.line(surf, PX["sep"], (pad, y), (bw - pad, y))
    y += 5

    key_labels = ["W", "A", "S", "D", "SPC", "SHF"]
    box_w_map = {"W": 34, "A": 34, "S": 34, "D": 34, "SPC": 48, "SHF": 48}
    box_h = 26
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

    pygame.draw.line(surf, PX["sep"], (pad, y), (bw - pad, y))
    y += 5
    help_lines = [
        "Mouse/RStick:look  Scroll/LB,RB:zoom",
        "WASD/LStick:move   SPC/RT:up  SHF/LT:dn",
        "Arrows/D-pad:torque  X:free/select/back",
        "V/B:view  L:lig style  P/A:pause",
        "F11/Y:fullscreen",
        "ESC/Start:quit",
    ]
    for line in help_lines:
        surf.blit(_txt(line, PX["dim"], big=False), (pad, y))
        y += 19

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


def draw_pause_menu(aw, ah, params, sel):
    pw, ph = 520, len(params) * 34 + 88
    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((6, 6, 20, 238))
    pygame.draw.rect(surf, (0, 180, 220), (0, 0, pw, ph), 2)

    y = 14
    title = _txt("PAUSED  [P] resume", (255, 200, 50))
    surf.blit(title, ((pw - title.get_width()) // 2, y))
    y += 38
    pygame.draw.line(surf, (40, 60, 80), (10, y), (pw - 10, y))
    y += 12

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
        surf.blit(_txt(line, col, big=False), (20, y))
        y += 34

    surf.blit(_txt("  Left/Right to adjust", (100, 100, 130), big=False),
              (20, y))

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
