#!/usr/bin/env python
"""
Compute per-frame SASA of inhibitor LIGAND from imaging-fixed trajectories.

Uses biotite Shrake-Rupley (all heavy atoms as context, reports only LIGAND SASA).
Reads the complex-only PSF + fixed DCDs produced by fix_imaging_inhibitor.py.

Usage:
    python compute_sasa_inhibitor.py <system> <rep>
    system : me1 or me3
    rep    : 0-4
"""

import sys
import os
import numpy as np

import MDAnalysis as mda
import biotite.structure as struc

system = sys.argv[1].lower()
rep    = int(sys.argv[2])

from config import BASE  # set ALLOSTERY_DATA to override
OUT  = f"{BASE}/dynamics/fixed_dcd"

if system == 'me1':
    DCD_PREFIX = 'me1_239_malate'
    INHIB_SEG  = 'LG00'
else:
    DCD_PREFIX = 'me3_239_malate'
    INHIB_SEG  = 'LIGA'

SEGMENTS    = [(i * 100, (i + 1) * 100) for i in range(10)]
COMPLEX_PSF = f"{OUT}/{DCD_PREFIX}_rep{rep}_complex.psf"
SASA_NPY    = f"{OUT}/{DCD_PREFIX}_rep{rep}_sasa.npy"

fixed_dcds = [
    f"{OUT}/{DCD_PREFIX}_{rep}_{s}to{e}ns_fixed.dcd"
    for (s, e) in SEGMENTS
]
fixed_dcds = [d for d in fixed_dcds if os.path.isfile(d)]

if not fixed_dcds:
    print(f"No fixed DCDs found for {system} rep {rep}. "
          f"Run fix_imaging_inhibitor.py first.", flush=True)
    sys.exit(1)

print(f"System: {system.upper()}  Replicate: {rep}", flush=True)
print(f"PSF: {COMPLEX_PSF}", flush=True)
print(f"DCDs: {len(fixed_dcds)} segments", flush=True)

u = mda.Universe(COMPLEX_PSF, fixed_dcds)

# Heavy atoms only (water already stripped; exclude H)
all_heavy = u.select_atoms('not name H* and not name h*')
inhib_ag  = u.select_atoms(f'segid {INHIB_SEG}')
inhib_heavy = u.select_atoms(f'segid {INHIB_SEG} and not name H* and not name h*')

print(f"Total heavy atoms: {len(all_heavy)}", flush=True)
print(f"Inhibitor heavy atoms: {len(inhib_heavy)}", flush=True)


def mda_to_biotite(ag):
    """Convert MDAnalysis AtomGroup to biotite AtomArray (current frame)."""
    n   = len(ag)
    arr = struc.AtomArray(n)
    arr.coord = ag.positions.copy()

    # Elements: MDAnalysis 2.x exposes .elements for PSF-loaded systems
    try:
        elems = np.array(ag.elements, dtype='U4')
        # Replace empty strings with best-guess from atom name
        for i, e in enumerate(elems):
            if not e.strip():
                name = ag.atoms[i].name.lstrip('0123456789')
                elems[i] = name[0].upper() if name else 'C'
    except Exception:
        elems = np.array(
            [a.name.lstrip('0123456789')[0].upper() for a in ag],
            dtype='U4'
        )
    arr.element   = elems
    arr.res_id    = ag.resids
    arr.res_name  = ag.resnames
    arr.atom_name = ag.names
    # biotite chain_id is single char; use first char of segid
    arr.chain_id  = np.array([s[0] if s else 'A' for s in ag.segids], dtype='U1')
    return arr


# Pre-build inhibitor mask on the heavy-atom group (same across all frames)
inhib_mask = np.isin(all_heavy.indices, inhib_heavy.indices)
print(f"Inhibitor mask: {inhib_mask.sum()} atoms", flush=True)

sasa_vals  = []
n_total    = len(u.trajectory)

for i, ts in enumerate(u.trajectory):
    arr = mda_to_biotite(all_heavy)

    # Compute SASA for inhibitor atoms in full protein context
    # vdw_radii='Single' uses per-element VdW radii (safe for small molecules)
    per_atom = struc.sasa(
        arr,
        probe_radius=1.4,
        atom_filter=inhib_mask,
        ignore_ions=False,
        point_number=200,
        vdw_radii='Single'
    )
    total_sasa = float(np.nansum(per_atom[inhib_mask]))
    sasa_vals.append(total_sasa)

    if (i + 1) % 2000 == 0:
        print(f"  Frame {i + 1}/{n_total}  SASA={total_sasa:.1f} Å²", flush=True)

sasa_arr = np.array(sasa_vals)
np.save(SASA_NPY, sasa_arr)

print(f"\nSaved SASA -> {SASA_NPY}  ({len(sasa_arr)} frames)", flush=True)
print(f"SASA: mean={sasa_arr.mean():.1f}  std={sasa_arr.std():.1f}  "
      f"min={sasa_arr.min():.1f}  max={sasa_arr.max():.1f} Å²", flush=True)
