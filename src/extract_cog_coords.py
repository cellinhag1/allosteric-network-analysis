#!/usr/bin/env python
"""
Extract PROA residue center-of-geometry (COG) coordinates from ensemble MD
trajectories (50 ns, up to 25 replicates). Mirrors extract_ca_coords.py but
uses the heavy-atom COG of each residue rather than Cα position.

Kabsch alignment to first frame of that replica. Inhibitor distance filter
applied for me1_inh and me3_inh conditions.

Usage:
    python extract_cog_coords.py <condition> <rep>

    condition : me1_inh | me1_noinh | me3_inh | me3_noinh
    rep       : 0-24

Output (in data_cog/):
    <condition>_rep<rep>_cog.npy        shape (n_kept, N_res, 3)  float32
    <condition>_rep<rep>_frame_idx.npy  shape (n_kept,)           int32
    <condition>_resids.npy              shape (N_res,)  saved once per condition
"""
import sys
import os
import numpy as np
import MDAnalysis as mda

EXPECTED_FRAMES = 10_000
DIST_CUTOFF     = 5.0

condition = sys.argv[1].lower()
rep       = int(sys.argv[2])

from config import BASE  # set ALLOSTERY_DATA to override
ALLOD = f'{BASE}/allostery'
DATA  = f'{ALLOD}/data_cog'
os.makedirs(DATA, exist_ok=True)

COND_MAP = {
    'me1_inh': {
        'subdir':  f'{ALLOD}/me1_inhibitor/dynamics',
        'psf':     f'{BASE}/me1_239_malate.psf',
        'pattern': 'me1_inh_{rep}_prod.dcd',
    },
    'me1_noinh': {
        'subdir':  f'{ALLOD}/me1_no_inhibitor/dynamics',
        'psf':     f'{BASE}/me1_malate_no_inhibitor.psf',
        'pattern': 'me1_noinh_{rep}_prod.dcd',
    },
    'me3_inh': {
        'subdir':  f'{ALLOD}/me3_inhibitor/dynamics',
        'psf':     f'{BASE}/me3_239_malate.psf',
        'pattern': 'me3_inh_{rep}_prod.dcd',
    },
    'me3_noinh': {
        'subdir':  f'{ALLOD}/me3_no_inhibitor/dynamics',
        'psf':     f'{BASE}/me3_malate_no_inhibitor.psf',
        'pattern': 'me3_noinh_{rep}_prod.dcd',
    },
}

INHIB_CFG = {
    'me1_inh': {'seg': 'LG00', 'pocket_resids': [260, 263, 321, 360, 485]},
    'me3_inh': {'seg': 'LIGA', 'pocket_resids': [285, 288, 346, 385, 510]},
}
has_inhib = condition in INHIB_CFG

cfg      = COND_MAP[condition]
psf_path = cfg['psf']
out_npy  = f"{DATA}/{condition}_rep{rep}_cog.npy"
out_idx  = f"{DATA}/{condition}_rep{rep}_frame_idx.npy"
out_ids  = f"{DATA}/{condition}_resids.npy"

# For reruns: prefer *_prod_rerun.dcd over *_prod.dcd if it exists.
# The rerun DCD replaces the original for reps that had inhibitor dissociation.
default_dcd = f"{cfg['subdir']}/{cfg['pattern'].format(rep=rep)}"
rerun_dcd   = default_dcd.replace('_prod.dcd', '_prod_rerun.dcd')
if os.path.isfile(rerun_dcd) and os.path.getsize(rerun_dcd) > 5_000_000:
    dcd_path = rerun_dcd
    # Remove stale COG files so extraction reruns with the new trajectory
    for stale in (out_npy, out_idx):
        if os.path.isfile(stale):
            os.remove(stale)
            print(f"  Removed stale COG file (rerun DCD detected): {stale}", flush=True)
else:
    dcd_path = default_dcd

print(f"[{condition} rep {rep}]", flush=True)
print(f"  PSF: {psf_path}", flush=True)
print(f"  DCD: {dcd_path}", flush=True)
if has_inhib:
    print(f"  Inhibitor distance filter: <= {DIST_CUTOFF} A "
          f"(seg={INHIB_CFG[condition]['seg']})", flush=True)

if not os.path.isfile(dcd_path):
    print("  ERROR: DCD not found.", flush=True)
    sys.exit(1)

if os.path.isfile(out_npy):
    print(f"  EXISTS: {out_npy} -- skipping", flush=True)
    sys.exit(0)

u       = mda.Universe(psf_path, dcd_path)
protein = u.select_atoms('segid PROA and not name H*')
residues = protein.residues
N_res   = len(residues)
resids  = np.array([r.resid for r in residues])
print(f"  Residues: {N_res}", flush=True)

if not os.path.isfile(out_ids):
    np.save(out_ids, resids)
    print(f"  Saved resids -> {out_ids}", flush=True)

if has_inhib:
    icfg      = INHIB_CFG[condition]
    inhib     = u.select_atoms(f"segid {icfg['seg']}")
    poc_sel   = ' or '.join(f'resid {r}' for r in icfg['pocket_resids'])
    pocket_ca = u.select_atoms(f"segid PROA and name CA and ({poc_sel})")
    print(f"  Inhibitor atoms: {len(inhib)}  "
          f"Pocket CA atoms: {len(pocket_ca)}", flush=True)

# Alignment reference: COG per residue at frame 0
u.trajectory[0]
ref_pos = np.array([r.atoms.center_of_geometry() for r in residues])
ref_com = ref_pos.mean(0)
ref_c   = ref_pos - ref_com

n_frames = len(u.trajectory)
if n_frames < EXPECTED_FRAMES:
    print(f"  WARNING: only {n_frames}/{EXPECTED_FRAMES} frames.", flush=True)


def kabsch_rot(P, Q):
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def cog_distance(pos_a, pos_b):
    return float(np.linalg.norm(pos_a.mean(0) - pos_b.mean(0)))


print(f"  Scanning {n_frames} frames ...", flush=True)
coord_list = []
idx_list   = []
n_skip     = 0

for i, ts in enumerate(u.trajectory):
    if has_inhib:
        if cog_distance(inhib.positions, pocket_ca.positions) > DIST_CUTOFF:
            n_skip += 1
            continue

    pos   = np.array([r.atoms.center_of_geometry() for r in residues])
    com   = pos.mean(0)
    pos_c = pos - com
    R     = kabsch_rot(pos_c, ref_c)
    coord_list.append((pos_c @ R.T + ref_com).astype(np.float32))
    idx_list.append(i)

    if (i + 1) % 2000 == 0:
        print(f"    Frame {i+1}/{n_frames}  kept={len(idx_list)}", flush=True)

n_kept = len(coord_list)
if n_kept == 0:
    print(f"  ERROR: 0 frames kept. Check inhibitor binding / DIST_CUTOFF.", flush=True)
    sys.exit(1)

coords    = np.stack(coord_list).astype(np.float32)
frame_idx = np.array(idx_list, dtype=np.int32)

np.save(out_npy, coords)
np.save(out_idx, frame_idx)

if has_inhib:
    pct = 100 * n_kept / n_frames
    print(f"  Bound frames: {n_kept}/{n_frames} ({pct:.1f}%)  "
          f"skipped={n_skip}", flush=True)
print(f"  Saved -> {out_npy}  shape={coords.shape}", flush=True)
print(f"  Saved -> {out_idx}  ({len(frame_idx)} indices)", flush=True)
