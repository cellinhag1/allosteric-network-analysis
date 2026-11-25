#!/usr/bin/env python
"""
Extract PROA Cα coordinates from ensemble MD trajectories.
Aligns each frame to the first frame of that replica (Kabsch algorithm)
to remove global translation/rotation -- fit computed using ONLY Domain
A+B atoms (resid 17-277 + 464-535), then applied to all atoms, so that
Domain C's own motion doesn't leak into the fit itself and dilute its
apparent mobility (matches the CHARMM hinge-angle analysis's Domain-A+B-
anchored approach; see extract_ca_coords_restrained.py for the full
rationale).

For inhibitor conditions (me1_inh, me3_inh): only frames where the inhibitor
COG is within DIST_CUTOFF Å of the allosteric pocket CA COG are kept.
Minimum image convention is applied for the distance calculation.

Usage:
    python extract_ca_coords.py <condition> <rep>

    condition : me1_inh | me1_noinh | me3_inh | me3_noinh
    rep       : 0-24

Output (in data/):
    <condition>_rep<rep>_ca.npy        shape (n_kept, N_ca, 3)  float32
    <condition>_rep<rep>_frame_idx.npy shape (n_kept,)          int32
        -- original trajectory frame indices; used by compute_dccm_mi.py
           to correctly apply the last-25ns time filter even after distance
           filtering has dropped some frames
    <condition>_resids.npy             shape (N_ca,)  saved once per condition
"""
import sys
import os
import numpy as np
import MDAnalysis as mda

EXPECTED_FRAMES = 10_000   # 50 ns at 5 ps/frame
DIST_CUTOFF     = 5.0      # Å — COG(inhibitor) to COG(pocket CA); frame excluded if beyond this

condition = sys.argv[1].lower()
rep       = int(sys.argv[2])

from config import BASE  # set ALLOSTERY_DATA to override
ALLOD = f'{BASE}/allostery'
DATA  = f'{ALLOD}/data'
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

# Inhibitor segment name and allosteric pocket CA resids for distance filtering
INHIB_CFG = {
    'me1_inh': {'seg': 'LG00', 'pocket_resids': [260, 263, 321, 360, 485]},
    'me3_inh': {'seg': 'LIGA', 'pocket_resids': [285, 288, 346, 385, 510]},
}
has_inhib = condition in INHIB_CFG

cfg      = COND_MAP[condition]
dcd_path = f"{cfg['subdir']}/{cfg['pattern'].format(rep=rep)}"
psf_path = cfg['psf']
out_npy  = f"{DATA}/{condition}_rep{rep}_ca.npy"
out_idx  = f"{DATA}/{condition}_rep{rep}_frame_idx.npy"
out_ids  = f"{DATA}/{condition}_resids.npy"

print(f"[{condition} rep {rep}]", flush=True)
print(f"  PSF: {psf_path}", flush=True)
print(f"  DCD: {dcd_path}", flush=True)
if has_inhib:
    print(f"  Inhibitor distance filter: ≤ {DIST_CUTOFF} Å  "
          f"(seg={INHIB_CFG[condition]['seg']})", flush=True)

if not os.path.isfile(dcd_path):
    print(f"  ERROR: DCD not found.", flush=True)
    sys.exit(1)

if os.path.isfile(out_npy):
    print(f"  EXISTS: {out_npy} -- skipping", flush=True)
    sys.exit(0)

u  = mda.Universe(psf_path, dcd_path)
ca = u.select_atoms('segid PROA and name CA')
print(f"  Cα atoms: {len(ca)}", flush=True)

if not os.path.isfile(out_ids):
    np.save(out_ids, ca.resids.copy())
    print(f"  Saved resids -> {out_ids}", flush=True)

# Inhibitor + pocket selections (only for inhibitor conditions)
if has_inhib:
    icfg      = INHIB_CFG[condition]
    inhib     = u.select_atoms(f"segid {icfg['seg']}")
    poc_sel   = ' or '.join(f'resid {r}' for r in icfg['pocket_resids'])
    pocket_ca = u.select_atoms(f"segid PROA and name CA and ({poc_sel})")
    print(f"  Inhibitor atoms: {len(inhib)}  "
          f"Pocket CA atoms: {len(pocket_ca)}", flush=True)

# Domain A+B: the fit/reference selection for Kabsch alignment (ME1 numbering,
# validated project-wide, matches the CHARMM hinge-angle Domain A+B selection).
# NOTE: these resid ranges are ME1-specific -- if this script is ever run for
# me3_inh/me3_noinh, verify ME3's domain boundaries match before trusting the fit.
DOMAIN_A, DOMAIN_B1, DOMAIN_B2 = (17, 130), (131, 277), (464, 535)


def domain_ab_mask(resids):
    return (((resids >= DOMAIN_A[0]) & (resids <= DOMAIN_A[1])) |
            ((resids >= DOMAIN_B1[0]) & (resids <= DOMAIN_B1[1])) |
            ((resids >= DOMAIN_B2[0]) & (resids <= DOMAIN_B2[1])))


# First frame as alignment reference for Kabsch
u.trajectory[0]
resids_all = ca.resids.copy()
ab_mask = domain_ab_mask(resids_all)
n_ab = int(ab_mask.sum())
ref_pos_full = ca.positions.copy()
ref_pos_ab = ref_pos_full[ab_mask]
ref_com_ab = ref_pos_ab.mean(0)
ref_c_ab = ref_pos_ab - ref_com_ab
print(f"  Domain A+B fit atoms: {n_ab}", flush=True)
assert n_ab == 333, f"Expected 333 Domain A+B fit atoms, got {n_ab}"

n_frames = len(u.trajectory)
if n_frames < EXPECTED_FRAMES:
    print(f"  WARNING: only {n_frames}/{EXPECTED_FRAMES} frames — "
          f"trajectory may be incomplete.", flush=True)


def kabsch_rot(P, Q):
    """Rotation matrix R such that P @ R ≈ Q (both Nx3, mean-centred)."""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R


def cog_distance(pos_a, pos_b):
    """Euclidean distance between centers of geometry of pos_a and pos_b."""
    return float(np.linalg.norm(pos_a.mean(0) - pos_b.mean(0)))


print(f"  Scanning {n_frames} frames ...", flush=True)
coord_list = []
idx_list   = []
n_skip     = 0

for i, ts in enumerate(u.trajectory):
    # Distance filter for inhibitor conditions
    if has_inhib:
        if cog_distance(inhib.positions, pocket_ca.positions) > DIST_CUTOFF:
            n_skip += 1
            continue

    # Kabsch alignment of Cα to first frame -- fit on Domain A+B only
    pos_full = ca.positions.copy()
    pos_ab   = pos_full[ab_mask]
    com_ab   = pos_ab.mean(0)
    pos_ab_c = pos_ab - com_ab
    R        = kabsch_rot(pos_ab_c, ref_c_ab)
    aligned_full = (pos_full - com_ab) @ R.T + ref_com_ab
    coord_list.append(aligned_full.astype(np.float32))
    idx_list.append(i)

    if (i + 1) % 2000 == 0:
        kept_so_far = len(idx_list)
        print(f"    Frame {i+1}/{n_frames}  kept={kept_so_far}", flush=True)

n_kept = len(coord_list)
if n_kept == 0:
    print(f"  ERROR: 0 frames passed the distance filter — "
          f"check inhibitor binding / DIST_CUTOFF ({DIST_CUTOFF} Å).", flush=True)
    sys.exit(1)

coords    = np.stack(coord_list).astype(np.float32)   # (n_kept, N_ca, 3)
frame_idx = np.array(idx_list, dtype=np.int32)        # (n_kept,)

np.save(out_npy, coords)
np.save(out_idx, frame_idx)

if has_inhib:
    pct = 100 * n_kept / n_frames
    print(f"  Bound frames: {n_kept}/{n_frames} ({pct:.1f}%)  "
          f"skipped={n_skip}", flush=True)
print(f"  Saved -> {out_npy}  shape={coords.shape}", flush=True)
print(f"  Saved -> {out_idx}  ({len(frame_idx)} indices)", flush=True)
