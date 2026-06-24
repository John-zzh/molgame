# MolGame

Real-time molecular dynamics game powered by [OpenMM](https://openmm.org/). Steer a probe atom, ligand, ion, or selected residue around a solvated protein with live GPU-backed physics.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![OpenMM 8.x](https://img.shields.io/badge/OpenMM-8.x-green)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow)

## Features

- **Real MD simulation** — Amber14SB force field + TIP3P explicit water, PME electrostatics, Langevin dynamics
- **Any PDB** — Load any protein from the RCSB by PDB ID, auto-downloaded on first run
- **Ligands and ions** — Control a GAFF2-parameterized ligand or ions initialized from the PDB
- **Multiple views** — Sticks, molecular surface, and C-alpha backbone views, switchable with `V`
- **Ligand styles** — Switch real ligands between CPK ball and line render modes with `L`
- **Live telemetry** — HUD shows total PE, clean PE, and a rolling clean-PE history graph
- **Snapshots** — Save the current protein/ligand structure from the pause menu, with optional water
- **Retro pixel UI** — NES/Contra-inspired HUD, scanline CRT overlay, proximity bar, hi-score tracking, pause menu
- **GPU accelerated** — CUDA/OpenCL/CPU OpenMM backend selection + OpenGL rendering

## Quick Start

### 1. Create environment

```bash
mamba env create -f environment.yml
mamba activate molgame
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate molgame
```

### 2. Run

```bash
python start.py
```

The default protein is **1UBQ** (Ubiquitin, 76 residues). To load a different protein:

```bash
python start.py --pdb 2LYZ   # Lysozyme
python start.py --pdb 1CRN   # Crambin
```

PDB files are automatically downloaded from RCSB on first run.

### Windows and WSL

MolGame can run from PowerShell, Anaconda Prompt, macOS/Linux shells, or WSL. Use the same conda/mamba environment on every platform:

```powershell
mamba env create -f environment.yml
mamba activate molgame
python start.py
```

On native Windows, `run.bat` is also available after activating the environment:

```powershell
run.bat --pdb 1UBQ
```

On WSL, a working GUI/OpenGL stack is required. Windows 11 WSLg usually works out of the box. Older WSL setups need an external X server and working `DISPLAY`/OpenGL forwarding. If the window does not open, first verify a simple pygame/OpenGL application can create a window inside WSL.

OpenMM backend selection is automatic by default. The resolver tries `CUDA`, then `OpenCL`, then `CPU`. You can force a backend when needed:

```bash
python start.py --platform CUDA
python start.py --platform OpenCL
python start.py --platform CPU
```

The same setting can be provided with an environment variable:

```bash
MOLGAME_OPENMM_PLATFORM=CPU python start.py
```

### Linux / Ubuntu notes

Native Ubuntu desktop is supported as long as the system can create a Pygame/OpenGL window:

```bash
mamba env create -f environment.yml
mamba activate molgame
python start.py
```

Ubuntu servers, SSH sessions, containers, and headless machines need extra GUI/OpenGL setup such as X11 forwarding, VirtualGL, or a desktop session. Without a working display, Pygame cannot open the interactive window even if OpenMM itself is installed correctly.

To check which OpenMM compute backends are visible on Ubuntu:

```bash
python -c "import openmm as mm; print([mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])"
```

For NVIDIA systems, `CUDA` is usually preferred. If the automatic resolver picks the wrong backend, force one with `--platform`.

### Protein + ligand

Use `--ligand` with the ligand residue name found in the PDB. The ligand is extracted, parameterized with GAFF2/AM1-BCC through OpenFF/openmmforcefields, then added back to the solvated system.

```bash
python start.py --pdb 4HJO --ligand AQ4
```

If the residue name is wrong or missing from the PDB, startup exits with a ligand-not-found error.

### Local protein and ligand files

Use `--pdb-file` to load a local protein PDB instead of downloading from RCSB. Use `--ligand-file` to add one or more local ligands from SDF or MOL2. Each ligand file should already contain explicit hydrogens and 3D coordinates aligned to the protein coordinate frame.

```bash
python start.py --pdb-file ./protein.pdb
python start.py --pdb-file ./protein.pdb --ligand-file ./ligand.sdf
python start.py --pdb-file ./protein.pdb --ligand-file ./ligand.mol2
python start.py --pdb-file ./protein.pdb --ligand-file ./lig1.sdf --ligand-file ./lig2.mol2
```

Repeated `--ligand-file` arguments are treated as separate ligand groups. The first ligand starts as the controlled target. Press `X` to enter free/select mode, aim at another ligand, then press `X` again to control that ligand group independently.

`--ligand` and `--ligand-file` are mutually exclusive. `--ion` cannot be combined with ligand control.

When running from another working directory, MolGame looks for `config.json` and `gaff_cache.json` in the current directory first. If they do not exist there, it falls back to the source checkout directory. Saved settings are written to the current working directory.

### Protein + ion

Use `--ion` to control one or more supported ions. Supported names are `CA`, `NA`, `CL`, `K`, `MG`, and `ZN`.

```bash
python start.py --pdb 4MS2 --ion CA
python start.py --pdb 4MS2 --ion NA
python start.py --pdb 4MS2 --ion MG
```

When multiple matching ions are present, the first starts as the active target. Press `X` from free-look mode to select another ion or residue.

## Controls

| Key | Action |
|-----|--------|
| `W` / `S` | Move forward / backward (camera-relative) |
| `A` / `D` | Move left / right |
| `Space` | Move up |
| `Shift` | Move down |
| Mouse | Look around |
| Arrow keys | Apply physical torque force to the selected target |
| Scroll | Zoom in / out |
| `X` | Cycle ligand/control target → free look → selected residue/ion |
| `V` | Cycle sticks / surface / backbone view |
| `L` | Toggle ligand ball / line style |
| `P` | Pause and open simulation/settings menu |
| `F11` | Toggle fullscreen |
| `ESC` | Quit |

Gamepads are also supported: left stick moves, right stick looks, triggers move vertically, D-pad applies torque, `A` pauses, `B` changes view, `X` selects, and `Y` toggles fullscreen.

When `X` enters free/select mode, the crosshair turns yellow and the ligand glow is hidden. This is the visual cue that the next `X` press will pick a new target.

The pause menu includes `Select scope`, which controls what `X` selects when aiming at protein atoms: `Residue` selects the hit residue, while `Chain` selects the entire protein chain containing the hit atom.

The pause menu also includes `Force mode`. `Total` divides the requested force across all selected atoms, while `Per atom` applies the requested force to each selected atom so large selections such as chains move more noticeably.

Real ligands can feel harder to move than selected protein residues because they are usually buried in a pocket and strongly coupled to surrounding atoms. `Lig force x` scales steering force only when controlling the original ligand target.

While paused, choose `Save PDB` and press `Y` on a gamepad or `Enter` on the keyboard to save the current structure. Keyboard `S` is also available as a shortcut. Files are written as `molgame_snapshot_YYYYMMDD_HHMMSS_dry.pdb` or `_all.pdb` in the current working directory. The `Save water` menu option controls whether waters are included; the default `No` saves protein and ligand only. Extra particles that are not part of the topology, such as the default probe atom, are not written.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Pygame      │────▶│  OpenMM      │────▶│  OpenGL     │
│  Input       │     │  MD Engine   │     │  Renderer   │
│  WASD/Mouse  │     │CUDA/OpenCL/CPU│    │  (GPU)      │
└─────────────┘     └──────────────┘     └─────────────┘
```

1. **Setup** — PDBFixer cleans the structure, Modeller adds TIP3P solvent, and the system is built with Amber14SB + PME. By default a single probe atom is injected near the protein surface. With `--ligand`, the ligand is parameterized with GAFF2 and steered as a molecule. With `--ion`, supported ions are injected from PDB coordinates and can be selected individually.

2. **Game loop** — Each frame:
   - WASD input is converted to a force vector in the camera's reference frame
   - The steering force is applied through `CustomExternalForce`
   - Arrow keys/D-pad apply a separate torque `CustomExternalForce` to the current multi-atom target
   - OpenMM advances the simulation with the configured MD steps per frame
   - HUD energy shows total PE, clean PE, and a rolling clean-PE graph; clean PE subtracts steering/torque force-group energy
   - Atom positions are read back and rendered with OpenGL

3. **Target selection** — `X` switches from direct target control to free-look selection. From there, aim at a residue, chain, or ion and press `X` again to steer it. Press `X` once more while steering a selected protein target to return to the original ligand/probe/ion target.

4. **Stability controls** — The pause menu can adjust force, force mode, ligand force scale, torque force, friction, temperature, MD steps per frame, timestep, C-alpha restraint strength, water draw radius, mouse look sensitivity, aim offset, ligand style, select scope, and crosshair style.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Force field | Amber14SB + TIP3P-FB |
| Electrostatics | PME, cutoff 0.9 nm |
| Integrator | LangevinMiddle |
| Timestep | Configurable, default 2 fs |
| Temperature | Configurable, default 300 K |
| Friction | Configurable, default 5 ps⁻¹ |
| MD steps/frame | Configurable, default 8 |
| Probe mass | 40 amu |
| Probe LJ | sigma=0.40 nm, epsilon=6.0 kJ/mol |
| Ligand force field | GAFF2 via openmmforcefields/OpenFF |
| Ion names | CA, NA, CL, K, MG, ZN |
| Backbone restraint | Configurable C-alpha restraint |

## Requirements

- macOS, Linux, Windows, or WSL with working OpenGL window support
- GPU with CUDA or OpenCL support recommended; CPU fallback is available but slower
- ~2 GB RAM for a typical small protein (~15k atoms with solvent)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
