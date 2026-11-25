#!/usr/bin/env python
"""
Open/closed-state + inhibitor-stability metrics for the completed 1us
restrained ME1 MD runs, per long_sim_allostery.txt.

Source data is ONLY dynamics_restrained_malate/pbc_fixed/ (water-stripped,
imaging-corrected) -- never the raw dcd/ or the full-system psf/crd.

Metrics computed per frame (all distances in Angstrom):
  malate_restraint      : resid179(CA) <-> malate C2   (the actual restrained
                           pair from dynamics_me1_restrained.py -- sanity
                           check that the flat-bottom NOE held, rmax=15A)
  inhib_pocket_dist      : LIGAND COM <-> pocket-CA centroid (with_inhibitor only)
                           pocket = resid 260,263,321,360,485 (LIGAND binding shell,
                           same definition as fix_imaging_inhibitor.py)
  lys183_malate          : LYS183(NZ) <-> malate hydroxyl O
                           NOTE: long_sim_allostery.txt calls this "malate C2
                           hydroxyl oxygen (O2)", but the actual RESI LMR
                           connectivity (me1_malate.str: BOND C2 O3, BOND O3 H2)
                           shows the hydroxyl on C2 is named O3, not O2 -- O2 is
                           one of C1's two carboxylate oxygens. Using O3 here,
                           which is the chemically correct hydroxyl atom.
  asp278_glu255          : min distance over ASP278{OD1,OD2} x GLU255{OE1,OE2}
                           ("closing lock")
  tyr112_malate_c3       : TYR112(OH) <-> malate C3
  arg165_malate          : min distance over ARG165{NH1,NH2} x malate C1-
                           carboxylate {O1,O2}
  asn418_malate          : min distance ASN418(ND2) x malate C1-carboxylate {O1,O2}
  mg_glu255 / mg_asp256 / mg_asp279 : Mg2+ <-> nearest carboxylate O of each
  mg_malate              : Mg2+ <-> nearest of malate {O1,O2,O3}
  ligand_ring1_phe263, ligand_ring1_his321,
  ligand_ring2_phe263, ligand_ring2_his321 : LIGAND ring-centroid <-> residue ring-
                           centroid (with_inhibitor only). LIGAND has two separate
                           aromatic systems (see RING1/RING2 below) -- both are
                           reported against both residues since which ring
                           actually stacks with which residue hasn't been
                           visually/structurally confirmed; NOT a claim that a
                           specific pairing is "the" stacking interaction.

Usage:
    python compute_openclosed_metrics.py <system> <rep>
    system : with_inhibitor | no_inhibitor
    rep    : 0-4
"""
import sys
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

system = sys.argv[1].lower()
rep    = int(sys.argv[2])

from config import DYN
FIXED = f"{DYN}/pbc_fixed"
from config import OUT
os.makedirs(OUT, exist_ok=True)

if system == 'with_inhibitor':
    PREFIX = 'me1_239_malate_restrained'
    HAS_INHIB = True
elif system == 'no_inhibitor':
    PREFIX = 'me1_malate_noinhibit_restrained'
    HAS_INHIB = False
else:
    sys.exit(f"Unknown system '{system}' -- expected with_inhibitor or no_inhibitor")

PSF = f"{FIXED}/{PREFIX}_complex.psf"
SEGMENTS = [(i * 100, (i + 1) * 100) for i in range(10)]
OUT_CSV = f"{OUT}/{PREFIX}_{rep}_metrics.csv"

if os.path.isfile(OUT_CSV):
    print(f"EXISTS, skipping: {OUT_CSV}")
    sys.exit(0)

POCKET_RESID = [260, 263, 321, 360, 485]

print(f"System: {system}  Replicate: {rep}", flush=True)

u = mda.Universe(PSF)

# ── Static atom-index selections (topology only, indices are frame-independent) ──
anchor_ca   = u.select_atoms('segid PROA and resid 179 and name CA')
malate_c2   = u.select_atoms('segid LG03 and name C2')
malate_o3   = u.select_atoms('segid LG03 and name O3')
malate_c3   = u.select_atoms('segid LG03 and name C3')
malate_c1_o = u.select_atoms('segid LG03 and name O1 O2')

lys183_nz    = u.select_atoms('segid PROA and resid 183 and name NZ')
asp278_od    = u.select_atoms('segid PROA and resid 278 and name OD1 OD2')
glu255_oe    = u.select_atoms('segid PROA and resid 255 and name OE1 OE2')
tyr112_oh    = u.select_atoms('segid PROA and resid 112 and name OH')
arg165_nh    = u.select_atoms('segid PROA and resid 165 and name NH1 NH2')
asn418_nd2   = u.select_atoms('segid PROA and resid 418 and name ND2')

mg           = u.select_atoms('segid IO00 and name MG')
glu255_oe_mg = u.select_atoms('segid PROA and resid 255 and name OE1 OE2')
asp256_od    = u.select_atoms('segid PROA and resid 256 and name OD1 OD2')
asp279_od    = u.select_atoms('segid PROA and resid 279 and name OD1 OD2')
malate_all_o = u.select_atoms('segid LG03 and name O1 O2 O3')

pocket_ca = u.select_atoms(
    'segid PROA and name CA and resid ' + ' '.join(str(r) for r in POCKET_RESID)
)

for name, ag, expect in [
    ('anchor_ca', anchor_ca, 1), ('malate_c2', malate_c2, 1),
    ('malate_o3', malate_o3, 1), ('malate_c3', malate_c3, 1),
    ('malate_c1_o', malate_c1_o, 2), ('lys183_nz', lys183_nz, 1),
    ('asp278_od', asp278_od, 2), ('glu255_oe', glu255_oe, 2),
    ('tyr112_oh', tyr112_oh, 1), ('arg165_nh', arg165_nh, 2),
    ('asn418_nd2', asn418_nd2, 1), ('mg', mg, 1),
    ('asp256_od', asp256_od, 2), ('asp279_od', asp279_od, 2),
    ('malate_all_o', malate_all_o, 3), ('pocket_ca', pocket_ca, len(POCKET_RESID)),
]:
    assert len(ag) == expect, f"Selection '{name}' expected {expect} atoms, got {len(ag)}"

if HAS_INHIB:
    inhib_all = u.select_atoms('segid LG00')
    # Two separate aromatic systems in LIGAND -- see module docstring.
    ligand_ring1 = u.select_atoms('segid LG00 and name C10 C11 C12 C13 C14 C21')
    ligand_ring2 = u.select_atoms('segid LG00 and name C5 C6 C7 C8 C9 N2 N3 N6 N7')
    phe263_ring = u.select_atoms('segid PROA and resid 263 and name CG CD1 CD2 CE1 CE2 CZ')
    his321_ring = u.select_atoms('segid PROA and resid 321 and name CG ND1 CE1 NE2 CD2')
    assert len(ligand_ring1) == 6 and len(ligand_ring2) == 9
    assert len(phe263_ring) == 6 and len(his321_ring) == 5

print(f"All selections validated.", flush=True)

rows = []

for (start, end) in SEGMENTS:
    in_dcd = f"{FIXED}/{PREFIX}_{rep}_{start}to{end}ns_fixed.dcd"
    if not os.path.isfile(in_dcd):
        print(f"  SKIP (missing): {in_dcd}", flush=True)
        continue

    print(f"Processing {start}-{end} ns ...", flush=True)
    u.load_new(in_dcd)
    n_frames = len(u.trajectory)

    for iframe, ts in enumerate(u.trajectory):
        t_ns = start + iframe / 100.0   # 100 frames/ns

        row = {
            'time_ns': t_ns,
            'malate_restraint': float(np.linalg.norm(
                anchor_ca.positions[0] - malate_c2.positions[0])),
            'lys183_malate': float(np.linalg.norm(
                lys183_nz.positions[0] - malate_o3.positions[0])),
            'asp278_glu255': float(distance_array(asp278_od.positions, glu255_oe.positions).min()),
            'tyr112_malate_c3': float(np.linalg.norm(
                tyr112_oh.positions[0] - malate_c3.positions[0])),
            'arg165_malate': float(distance_array(arg165_nh.positions, malate_c1_o.positions).min()),
            'asn418_malate': float(distance_array(asn418_nd2.positions, malate_c1_o.positions).min()),
            'mg_glu255': float(distance_array(mg.positions, glu255_oe_mg.positions).min()),
            'mg_asp256': float(distance_array(mg.positions, asp256_od.positions).min()),
            'mg_asp279': float(distance_array(mg.positions, asp279_od.positions).min()),
            'mg_malate': float(distance_array(mg.positions, malate_all_o.positions).min()),
        }

        if HAS_INHIB:
            row['inhib_pocket_dist'] = float(np.linalg.norm(
                inhib_all.positions.mean(axis=0) - pocket_ca.positions.mean(axis=0)))
            r1c = ligand_ring1.positions.mean(axis=0)
            r2c = ligand_ring2.positions.mean(axis=0)
            phec = phe263_ring.positions.mean(axis=0)
            hisc = his321_ring.positions.mean(axis=0)
            row['ligand_ring1_phe263'] = float(np.linalg.norm(r1c - phec))
            row['ligand_ring1_his321'] = float(np.linalg.norm(r1c - hisc))
            row['ligand_ring2_phe263'] = float(np.linalg.norm(r2c - phec))
            row['ligand_ring2_his321'] = float(np.linalg.norm(r2c - hisc))

        rows.append(row)

    if (start // 100) % 2 == 1:
        print(f"  ... {len(rows)} frames accumulated so far", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} frames -> {OUT_CSV}", flush=True)
