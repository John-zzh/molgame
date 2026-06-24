import os, json

W, H        = 1280, 720
FPS_CAP     = 60
CONTACT_CUT = 0.5

DEFAULTS = {
    "md_steps":       8,
    "timestep_fs":    2.0,
    "force":          500.0,
    "force_mode":     1,
    "ligand_force_scale": 5.0,
    "torque_force":   100.0,
    "friction":       5.0,
    "temperature":    300.0,
    "restraint_k":    0.0,
    "mouse_sens":     0.15,
    "scroll_sens":    0.5,
    "pad_look_sens":  5.0,
    "pad_zoom_sens":  0.08,
    "pad_deadzone":   0.15,
    "water_radius":   1.5,
    "aim_offset_y":   1.4,
    "ligand_style":   0,
    "select_scope":   0,
    "save_water":     0,
    "cross_size":     12,
    "cross_color":    0,
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def config_read_path():
    cwd_path = os.path.abspath("config.json")
    if os.path.exists(cwd_path):
        return cwd_path
    return SOURCE_CONFIG_PATH


def config_write_path():
    return os.path.abspath("config.json")


# ── Color helpers ──────────────────────────────────────────

def hex3(s):
    """'#RRGGBB' → (r, g, b) GL floats 0‒1"""
    h = s.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def hex4(s):
    """'#RRGGBBAA' → (r, g, b, a) GL floats 0‒1"""
    h = s.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6))

def hexP(s):
    """'#RRGGBB' or '#RRGGBBAA' → pygame int tuple"""
    h = s.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in range(0, len(h), 2))


# ── Atom colors (protein) ─────────────────────────────────

CPK = {k: hex3(v) for k, v in {
    "C":  "#525252",
    "N":  "#2433A6",
    "O":  "#A62424",
    "S":  "#A69924",
    "H":  "#D9D9D9",
}.items()}

CPK_TOON = {k: hex3(v) for k, v in {
    "C":  "#8C8C8C",
    "N":  "#4D73F2",
    "O":  "#F24D4D",
    "S":  "#E6CC40",
    "H":  "#EBEBF2",
}.items()}


# ── Atom colors (ligand) ──────────────────────────────────

LIG_CPK = {k: hex3(v) for k, v in {
    "C":  "#E68A2E",
    "N":  "#2D64FF",
    "O":  "#E53935",
    "S":  "#F2D34B",
    "H":  "#F0C890",
    "F":  "#35D982",
    "Cl": "#35D982",
    "Br": "#A64A2A",
    "P":  "#FF9F2E",
}.items()}


# ── VDW radii ─────────────────────────────────────────────

VDW_RENDER = {"C": 0.09, "N": 0.08, "O": 0.08, "S": 0.10, "H": 0.05}
VDW_REAL   = {"C": 0.170, "N": 0.155, "O": 0.152, "S": 0.180}


# ── 3D scene colors ───────────────────────────────────────

CL = {
    "bg":           hex3("#04040A"),
    "surface":      hex3("#476B94"),
    "water":        hex4("#6699EB4D"),
    "grid":         hex3("#122433"),
    "box":          hex3("#265973"),
    "crosshair":    hex4("#FFFFFF80"),
    "scanline":     hex4("#0000000E"),
    "probe_solid":  hex3("#26D9F2"),
    "probe_glow":   hex4("#1AB3E64D"),
    "lig_glow":     hex4("#1AB3E60A"),
    "ion_solid":    hex3("#66FF66"),
    "ion_glow":     hex4("#33FF664D"),
    "toon_outline": hex3("#000000"),
    "backbone":     hex3("#00D2FF"),
    "backbone_dim": hex3("#004B66"),
    "sel_glow":     hex4("#FFD93326"),
    "sel_atom":     hex4("#FFE066B3"),
    "axis_x":       hex3("#E63333"),
    "axis_y":       hex3("#33E633"),
    "axis_z":       hex3("#4D66E6"),
}

CROSS_COLORS = [
    hex4("#FFFFFF80"),  # White
    hex4("#33FF5580"),  # Green
    hex4("#FFEE3380"),  # Yellow
    hex4("#FF333380"),  # Red
]
CROSS_COLOR_NAMES = ["White", "Green", "Yellow", "Red"]


# ── HUD / pixel palette ───────────────────────────────────

PX = {
    "bg":      hexP("#060610D7"),
    "border":  hexP("#00D2FF"),
    "border2": hexP("#00648C"),
    "title":   hexP("#FFB914"),
    "stage":   hexP("#00FFD2"),
    "text":    hexP("#D2D2E1"),
    "dim":     hexP("#50506E"),
    "bar_lo":  hexP("#00C85A"),
    "bar_mid": hexP("#E6D200"),
    "bar_hi":  hexP("#FF3C1E"),
    "bar_bg":  hexP("#1A1A26"),
    "key_on":  hexP("#00D250"),
    "key_off": hexP("#202030"),
    "ok":      hexP("#32FF82"),
    "warn":    hexP("#FFFF3C"),
    "alert":   hexP("#FF3232"),
    "sep":     hexP("#243448"),
}


# ── Config I/O ─────────────────────────────────────────────

def load_config():
    cfg = dict(DEFAULTS)
    path = config_read_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            cfg.update({k: saved[k] for k in saved if k in DEFAULTS})
            print(f"Config loaded from {path}")
        except Exception as e:
            print(f"Warning: could not load config: {e}")
    return cfg


def save_config(cfg):
    path = config_write_path()
    try:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Config saved to {path}")
    except Exception as e:
        print(f"Warning: could not save config: {e}")
