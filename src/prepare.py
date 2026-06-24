import sys, os, time, tempfile
import numpy as np
import openmm as mm
from openmm import unit
import openmm.app as app
from pdbfixer import PDBFixer

from .config import VDW_REAL

ION_PARAMS = {
    "CA":  {"charge": 2.0, "mass": 40.08,  "sigma": 0.2413, "epsilon": 1.8874},
    "NA":  {"charge": 1.0, "mass": 22.99,  "sigma": 0.2439, "epsilon": 0.3658},
    "CL":  {"charge":-1.0, "mass": 35.45,  "sigma": 0.4478, "epsilon": 0.1489},
    "K":   {"charge": 1.0, "mass": 39.10,  "sigma": 0.3038, "epsilon": 0.3640},
    "MG":  {"charge": 2.0, "mass": 24.31,  "sigma": 0.1480, "epsilon": 3.6610},
    "ZN":  {"charge": 2.0, "mass": 65.38,  "sigma": 0.1949, "epsilon": 0.5230},
}

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}

WATER_RESIDUES = {"HOH", "WAT"}


def openmm_platform_names():
    return [mm.Platform.getPlatform(i).getName()
            for i in range(mm.Platform.getNumPlatforms())]


def choose_openmm_platform(requested="auto"):
    requested = (requested or os.environ.get("MOLGAME_OPENMM_PLATFORM")
                 or "auto").strip()
    available = openmm_platform_names()
    by_lower = {name.lower(): name for name in available}

    if requested.lower() != "auto":
        name = by_lower.get(requested.lower())
        if name is None:
            print(f"Warning: OpenMM platform '{requested}' not available; "
                  f"available: {', '.join(available)}")
        else:
            return mm.Platform.getPlatformByName(name), name

    for name in ("CUDA", "OpenCL", "CPU", "Reference"):
        if name.lower() in by_lower:
            actual = by_lower[name.lower()]
            return mm.Platform.getPlatformByName(actual), actual

    raise RuntimeError("No OpenMM platform available")


def gaff_cache_path():
    cwd_path = os.path.abspath("gaff_cache.json")
    source_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "gaff_cache.json")
    if os.path.exists(cwd_path):
        return cwd_path
    if os.path.exists(source_path):
        return source_path
    return cwd_path


def remove_incomplete_standard_residues(modeller):
    to_delete = []
    reports = []
    required = {"N", "CA", "C", "O"}
    for res in modeller.topology.residues():
        if res.name not in STANDARD_RESIDUES:
            continue
        atom_names = {atom.name for atom in res.atoms()}
        missing = sorted(required - atom_names)
        if missing:
            to_delete.append(res)
            chain = res.chain.id if res.chain.id else str(res.chain.index)
            reports.append(f"{res.name} {chain}:{res.id} missing {','.join(missing)}")
    if to_delete:
        modeller.delete(to_delete)
        print(f"     Removed {len(to_delete)} incomplete residue(s): "
              + "; ".join(reports[:6]))
        if len(reports) > 6:
            print(f"     ... and {len(reports) - 6} more")


def compute_surface(atom_pos, elem_list, probe_r=0.14, spacing=0.12):
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


def extract_ligand(pdb_path, lig_name):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("HETATM", "ATOM")) and line[17:20].strip() == lig_name:
                lines.append(line)
    if not lines:
        print(f"ERROR: Ligand '{lig_name}' not found in {pdb_path}")
        sys.exit(1)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False)
    for line in lines:
        tmp.write(line)
    tmp.write("END\n")
    tmp.close()

    mol = Chem.MolFromPDBFile(tmp.name, removeHs=False, sanitize=False)
    os.unlink(tmp.name)
    if mol is None:
        print(f"ERROR: RDKit could not parse ligand '{lig_name}'")
        sys.exit(1)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    mol = Chem.AddHs(mol, addCoords=True)
    print(f"     Ligand {lig_name}: {mol.GetNumHeavyAtoms()} heavy + "
          f"{mol.GetNumAtoms() - mol.GetNumHeavyAtoms()} H = {mol.GetNumAtoms()} atoms")
    return mol


def load_ligand_file(ligand_file):
    from rdkit import Chem

    ext = os.path.splitext(ligand_file)[1].lower()
    if ext in (".sdf", ".sd"):
        supplier = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False)
        mol = next((m for m in supplier if m is not None), None)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(ligand_file, removeHs=False, sanitize=False)
    else:
        print("ERROR: --ligand-file must be .sdf or .mol2")
        sys.exit(1)

    if mol is None:
        print(f"ERROR: RDKit could not parse ligand file '{ligand_file}'")
        sys.exit(1)
    if mol.GetNumConformers() == 0:
        print(f"ERROR: Ligand file '{ligand_file}' has no 3D coordinates")
        sys.exit(1)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    n_h = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
    if n_h == 0:
        print("Warning: ligand file contains no explicit hydrogens")
    print(f"     Ligand file {os.path.basename(ligand_file)}: "
          f"{mol.GetNumHeavyAtoms()} heavy + {n_h} H = {mol.GetNumAtoms()} atoms")
    print("     Local ligand hydrogens/coordinates preserved; no hydrogens added")
    return mol


def extract_ion(pdb_path, ion_name):
    positions = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("HETATM", "ATOM")) and line[17:20].strip() == ion_name:
                x = float(line[30:38]) / 10.0
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
                positions.append(np.array([x, y, z]))
    if not positions:
        print(f"ERROR: Ion '{ion_name}' not found in {pdb_path}")
        sys.exit(1)
    print(f"     Found {len(positions)} {ion_name} ion(s)")
    return positions


def prepare(pdb_path, cfg, lig_name=None, ion_name=None, ligand_file=None,
            openmm_platform="auto"):
    t0 = time.time()
    has_ligand = lig_name is not None or ligand_file is not None
    ligand_label = lig_name if lig_name else (
        os.path.basename(ligand_file) if ligand_file else None)
    step, nsteps = 0, 8 if has_ligand else 7

    ion_positions = None
    if ion_name:
        ion_positions = extract_ion(pdb_path, ion_name)

    # ── Extract & parameterize ligand ──
    if has_ligand:
        step += 1
        print(f"[{step}/{nsteps}] Preparing ligand {ligand_label} (GAFF2) …")
        from openff.toolkit import Molecule as OFFMolecule
        from openmmforcefields.generators import GAFFTemplateGenerator

        if ligand_file:
            rdkit_mol = load_ligand_file(ligand_file)
        else:
            rdkit_mol = extract_ligand(pdb_path, lig_name)
        off_mol = OFFMolecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
        cache_path = gaff_cache_path()
        print(f"     GAFF cache: {cache_path}")
        gaff = GAFFTemplateGenerator(molecules=off_mol, forcefield="gaff-2.11",
                                     cache=cache_path)
        lig_top = off_mol.to_topology().to_openmm()
        lig_pos_q = off_mol.conformers[0].to_openmm()
        lig_pos_nm = np.array(lig_pos_q.value_in_unit(unit.nanometers))
        lig_center = lig_pos_nm.mean(axis=0)

    # ── Fix protein ──
    step += 1
    print(f"[{step}/{nsteps}] Fixing PDB …")
    fixer = PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(keepWater=not ligand_file)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    # ── ForceField ──
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    if has_ligand:
        ff.registerTemplateGenerator(gaff.generator)

    # ── Solvate (protein only at this point) ──
    step += 1
    print(f"[{step}/{nsteps}] Solvating with TIP3P …")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    remove_incomplete_standard_residues(modeller)
    modeller.addSolvent(ff, model="tip3p", padding=0.8 * unit.nanometers,
                        ionicStrength=0.15 * unit.molar)

    pos = np.array(modeller.positions.value_in_unit(unit.nanometers))

    if not has_ligand:
        if ion_positions:
            lig_center = np.mean(ion_positions, axis=0)
        else:
            prot_idx_tmp = [a.index for a in modeller.topology.atoms()
                            if a.residue.name not in ("HOH", "WAT")]
            prot_pos = pos[prot_idx_tmp]
            pc = prot_pos.mean(axis=0)
            dx = prot_pos[:, 0] - pc[0]
            surf = prot_pos[np.argmax(dx)]
            direction = surf - pc
            direction /= np.linalg.norm(direction)
            lig_center = surf + direction * 1.0

    # ── Clear water overlap at ligand site ──
    step += 1
    print(f"[{step}/{nsteps}] Clearing water overlap …")
    to_delete = []
    if has_ligand:
        for res in modeller.topology.residues():
            if res.name in ("HOH", "WAT"):
                for atom in res.atoms():
                    if atom.element.symbol == "O":
                        d = np.min(np.linalg.norm(lig_pos_nm - pos[atom.index], axis=1))
                        if d < 0.25:
                            to_delete.append(res)
                        break
    else:
        ion_arr = np.array(ion_positions) if ion_positions else None
        for res in modeller.topology.residues():
            if res.name in ("HOH", "WAT"):
                for atom in res.atoms():
                    if atom.element.symbol == "O":
                        wp = pos[atom.index]
                        if ion_arr is not None:
                            if np.min(np.linalg.norm(ion_arr - wp, axis=1)) < 0.4:
                                to_delete.append(res)
                        elif np.linalg.norm(wp - lig_center) < 0.4:
                            to_delete.append(res)
                        break
    if to_delete:
        modeller.delete(to_delete)
        print(f"     Removed {len(to_delete)} water molecules")

    # ── Add ligand AFTER water deletion so indices are stable ──
    if has_ligand:
        step += 1
        print(f"[{step}/{nsteps}] Adding ligand to topology …")
        n_before_lig = sum(1 for _ in modeller.topology.atoms())
        modeller.add(lig_top, lig_pos_q)
        n_lig_atoms = off_mol.n_atoms
        lig_set = set(range(n_before_lig, n_before_lig + n_lig_atoms))

    # ── Collect atom indices ──
    pos = np.array(modeller.positions.value_in_unit(unit.nanometers))
    prot_idx, prot_elem, water_o, ca_idx = [], [], [], []
    ca_traces = {}
    lig_indices, lig_elem = [], []
    atom_records = []
    for atom in modeller.topology.atoms():
        res = atom.residue
        chain_id = res.chain.id.strip()[:1] if res.chain.id else chr(65 + (res.chain.index % 26))
        atom_records.append({
            "index": atom.index,
            "name": atom.name,
            "element": atom.element.symbol if atom.element is not None else "",
            "resname": res.name,
            "resid": res.id,
            "chain": chain_id,
            "is_water": res.name in WATER_RESIDUES,
            "is_protein": res.name in STANDARD_RESIDUES,
        })
        if atom.residue.name in WATER_RESIDUES:
            if atom.element.symbol == "O":
                water_o.append(atom.index)
        elif has_ligand and atom.index in lig_set:
            lig_indices.append(atom.index)
            lig_elem.append(atom.element.symbol)
        else:
            prot_idx.append(atom.index)
            prot_elem.append(atom.element.symbol)
            if atom.name == "CA":
                ca_idx.append(atom.index)
                ca_traces.setdefault(atom.residue.chain.index, []).append(atom.index)

    prot_center = pos[prot_idx].mean(axis=0) if prot_idx else pos.mean(axis=0)

    # ── Bonds (protein + ligand separately) ──
    prot_set = set(prot_idx)
    prot_bonds = []
    lig_bonds = []
    for bond in modeller.topology.bonds():
        a, b = bond[0].index, bond[1].index
        if has_ligand and a in lig_set and b in lig_set:
            lig_bonds.append((a, b))
        elif a in prot_set and b in prot_set:
            prot_bonds.append((a, b))

    if has_ligand:
        print(f"     Protein atoms: {len(prot_idx)},  Ligand atoms: {len(lig_indices)},  "
              f"Bonds: {len(prot_bonds)}+{len(lig_bonds)},  Water: {len(water_o)}")
    else:
        print(f"     Protein atoms: {len(prot_idx)},  Bonds: {len(prot_bonds)},  Water: {len(water_o)}")

    # ── Build system ──
    step += 1
    if has_ligand:
        print(f"[{step}/{nsteps}] Building force field (GAFF2 AM1-BCC, may take ~60s) …")
    else:
        print(f"[{step}/{nsteps}] Building force field …")
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometers, constraints=app.HBonds)

    rst = mm.CustomExternalForce("rst_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rst.addGlobalParameter("rst_k", cfg["restraint_k"])
    rst.addPerParticleParameter("x0")
    rst.addPerParticleParameter("y0")
    rst.addPerParticleParameter("z0")
    for idx in ca_idx:
        rst.addParticle(idx, pos[idx].tolist())
    system.addForce(rst)

    # ── Steering force (per-particle active flag for target switching) ──
    pf = mm.CustomExternalForce("active * -(fx*x + fy*y + fz*z)")
    pf.addGlobalParameter("fx", 0.0)
    pf.addGlobalParameter("fy", 0.0)
    pf.addGlobalParameter("fz", 0.0)
    pf.addPerParticleParameter("active")
    pf.setForceGroup(31)

    pf_map = {}
    tf = mm.CustomExternalForce("-(tx*x + ty*y + tz*z)")
    tf.addPerParticleParameter("tx")
    tf.addPerParticleParameter("ty")
    tf.addPerParticleParameter("tz")
    tf.setForceGroup(31)

    tf_map = {}
    for idx in prot_idx:
        pf_map[idx] = pf.addParticle(idx, [0.0])
        tf_map[idx] = tf.addParticle(idx, [0.0, 0.0, 0.0])

    if has_ligand:
        for idx in lig_indices:
            pf_map[idx] = pf.addParticle(idx, [1.0])
            tf_map[idx] = tf.addParticle(idx, [0.0, 0.0, 0.0])
        ligand = np.array(lig_indices)
        all_pos = pos
    else:
        nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
        ip = ION_PARAMS.get(ion_name) if ion_name else None
        if ion_positions and ip:
            ion_indices = []
            for i, ipos in enumerate(ion_positions):
                idx = system.addParticle(ip["mass"])
                nb.addParticle(ip["charge"], ip["sigma"], ip["epsilon"])
                pf_map[idx] = pf.addParticle(idx, [1.0 if i == 0 else 0.0])
                tf_map[idx] = tf.addParticle(idx, [0.0, 0.0, 0.0])
                ion_indices.append(idx)
            all_pos = np.vstack([pos] + [p.reshape(1, 3) for p in ion_positions])
            ligand = np.array(ion_indices)
            print(f"     Added {len(ion_indices)} {ion_name} ion(s), steering #{1}")
        else:
            lig_idx = system.addParticle(40.0)
            nb.addParticle(0.0, 0.40, 6.0)
            pf_map[lig_idx] = pf.addParticle(lig_idx, [1.0])
            tf_map[lig_idx] = tf.addParticle(lig_idx, [0.0, 0.0, 0.0])
            all_pos = np.vstack([pos, [lig_center]])
            ligand = np.array([lig_idx])
        lig_elem = []
        lig_bonds = []
    system.addForce(pf)
    system.addForce(tf)

    # ── Residue mapping (for X-key target switching) ──
    res_atoms = {}
    chain_atoms = {}
    for atom in modeller.topology.atoms():
        if atom.residue.name in ("HOH", "WAT"):
            continue
        if has_ligand and atom.index in lig_set:
            continue
        ckey = atom.residue.chain.index
        rkey = (ckey, atom.residue.id)
        res_atoms.setdefault(rkey, []).append(atom.index)
        chain_atoms.setdefault(ckey, []).append(atom.index)

    print(f"     Total particles: {system.getNumParticles()}")

    # ── Molecular surface (protein heavy atoms only) ──
    step += 1
    print(f"[{step}/{nsteps}] Computing molecular surface …")
    heavy_mask = np.array([e != "H" for e in prot_elem])
    heavy_pos = pos[np.array(prot_idx)[heavy_mask]]
    heavy_elem = [e for e in prot_elem if e != "H"]
    surface_data = compute_surface(heavy_pos, heavy_elem)

    # ── Simulation ──
    step += 1
    platform, platform_name = choose_openmm_platform(openmm_platform)
    print(f"[{step}/{nsteps}] Creating simulation ({platform_name}) …")
    timestep = cfg["timestep_fs"] * unit.femtoseconds
    integrator = mm.LangevinMiddleIntegrator(
        cfg["temperature"] * unit.kelvin, cfg["friction"] / unit.picosecond, timestep)
    print(f"      Timestep: {cfg['timestep_fs']:.2f} fs")
    ctx = mm.Context(system, integrator, platform)
    ctx.setPositions(all_pos * unit.nanometers)
    print("      Minimizing …")
    if ion_name:
        mm.LocalEnergyMinimizer.minimize(ctx, tolerance=1.0, maxIterations=5000)
    else:
        max_iter = 2000 if has_ligand else 500
        mm.LocalEnergyMinimizer.minimize(ctx, tolerance=10.0, maxIterations=max_iter)
    ctx.setVelocitiesToTemperature(cfg["temperature"] * unit.kelvin)
    if ion_name:
        print("      Equilibrating (2000 steps) …")
        integrator.step(2000)
    else:
        print("      Equilibrating …")
        integrator.step(500)
    box = ctx.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    box_origin = pos.min(axis=0)

    print(f"      Box: {box[0][0]:.2f} x {box[1][1]:.2f} x {box[2][2]:.2f} nm")
    print(f"      Ready in {time.time()-t0:.1f}s\n")
    return (ctx, integrator,
            np.array(prot_idx), np.array(prot_elem),
            np.array(water_o), ligand, prot_center, surface_data,
            box_origin, np.diag(box), prot_bonds,
            np.array(lig_elem), lig_bonds,
            pf, pf_map, tf, tf_map, res_atoms, chain_atoms,
            [np.array(trace, dtype=np.int32) for trace in ca_traces.values()],
            atom_records)
