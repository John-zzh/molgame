# MolGame

Real-time molecular dynamics game powered by [OpenMM](https://openmm.org/). Steer a probe atom around a solvated protein with WASD controls, while the physics simulation runs live on your GPU.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![OpenMM 8.x](https://img.shields.io/badge/OpenMM-8.x-green)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow)

## Features

- **Real MD simulation** — Amber14SB force field + TIP3P explicit water, PME electrostatics, Langevin dynamics at 300 K
- **Any PDB** — Load any protein from the RCSB by PDB ID (auto-downloaded)
- **Molecular surface** — Gaussian density + marching cubes surface rendering, toggle with `V` to atom spheres
- **Retro pixel UI** — NES/Contra-inspired HUD with scanline CRT overlay, proximity bar, hi-score tracking
- **GPU accelerated** — OpenCL for MD (OpenMM) + OpenGL for rendering, runs at 60 fps

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
python molgame.py
```

The default protein is **1UBQ** (Ubiquitin, 76 residues). To load a different protein:

```bash
python molgame.py --pdb 2LYZ   # Lysozyme
python molgame.py --pdb 1CRN   # Crambin
```

PDB files are automatically downloaded from RCSB on first run.

## Controls

| Key | Action |
|-----|--------|
| `W` / `S` | Move forward / backward (camera-relative) |
| `A` / `D` | Move left / right |
| `Space` | Move up |
| `Shift` | Move down |
| Mouse | Look around |
| Scroll | Zoom in / out |
| `V` | Toggle surface / atom view |
| `ESC` | Quit |

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Pygame      │────▶│  OpenMM      │────▶│  OpenGL     │
│  Input       │     │  MD Engine   │     │  Renderer   │
│  WASD/Mouse  │     │  (OpenCL GPU)│     │  (GPU)      │
└─────────────┘     └──────────────┘     └─────────────┘
```

1. **Setup** — PDBFixer cleans the structure, Modeller adds TIP3P solvent, the system is built with Amber14SB + PME. A single probe atom (argon-like, sigma=0.40 nm, epsilon=6.0 kJ/mol) is injected near the protein surface.

2. **Game loop** — Each frame:
   - WASD input is converted to a force vector in the camera's reference frame
   - The force is applied to the probe atom via `CustomExternalForce`
   - OpenMM steps the simulation (15 steps × 4 fs = 60 fs per frame)
   - Atom positions are read back and rendered with OpenGL

3. **Protein stability** — C-alpha atoms are restrained (k=1000 kJ/mol/nm²) so the backbone stays rigid while side chains and water move freely.

4. **Collision** — The probe has strong LJ parameters and the integrator uses high friction (5/ps), preventing it from tunneling through the protein.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Force field | Amber14SB + TIP3P-FB |
| Electrostatics | PME, cutoff 0.9 nm |
| Integrator | LangevinMiddle, 300 K |
| Timestep | 4 fs (HBonds constrained) |
| Friction | 5 ps⁻¹ |
| MD steps/frame | 15 |
| Probe mass | 40 amu |
| Probe LJ | sigma=0.40 nm, epsilon=6.0 kJ/mol |
| Backbone restraint | k=1000 kJ/mol/nm² on Cα |

## Requirements

- macOS or Linux (tested on macOS with Apple Silicon)
- GPU with OpenCL support
- ~2 GB RAM for a typical small protein (~15k atoms with solvent)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
