#!/usr/bin/env python
"""
Domain B/A superposition + Domain C rotation angle -- the literature's
"8 degree shift" metric from long_sim_allostery.txt.

Domain boundaries (validated against this exact construct's numbering --
residues 168-173 spell the conserved GLGDLG motif exactly as expected,
confirming zero offset needed):
    Domain A: 17-130     (N-terminal, construct start to Domain B)
    Domain B: 131-277 and 464-535   (two segments, sandwiching Domain C)
    Domain C: 278-463    ("inserted cassette", Rossmann/NAD(P)-binding domain)
Source: ME1-specific boundaries from Yoshida et al. 2022 (PDB 7X11/7X12,
the ME1 allosteric-inhibitor discovery paper -- almost certainly the
original source of the "8 degree" claim this project has been trying to
test), cross-checked against Xu et al. 1999 / Tao et al. 2003 (ME2).

Method: at each frame,
  1. Rigid-body superpose Domain A+B CA atoms onto their pose in the
     reference (starting) structure via the Kabsch algorithm -- this
     defines a fixed frame based on the (presumed more rigid) A+B core.
  2. Apply that same rotation to Domain C's CA atoms.
  3. Separately superpose the now-transformed Domain C onto its reference
     pose to get a second rotation matrix R_C -- this is how much
     Domain C has rotated *relative to the A+B frame*, isolating the
     inter-domain rotation from whatever the whole complex is doing.
  4. Convert R_C to a single rotation angle via the standard formula
     angle = arccos((trace(R_C) - 1) / 2).

This angle is 0 by construction at the reference frame and grows as
Domain C rotates away from its starting orientation relative to Domain
A+B -- directly comparable to the literature's "~8 degree rotation
required to transition from open to closed."

ALSO computed: a second, simpler metric --
`domain_bc_com_vector_angle_deg` -- the literature's originally-proposed
"vector-based angle between the centers of mass of Domain B and Domain C"
(background.txt), as a cross-check against the more rigorous rotation-
matrix method above. This one really is a plain angle-between-two-vectors
via the cosine formula (unlike the rotation-angle-from-a-matrix-trace
method, which only looks similar to cosine similarity -- see the
conversation this script came out of): after applying the SAME Domain-A+B
alignment (R_ab, already computed for the method above) to Domain B's own
coordinates too, form the vector from Domain B's center of mass to Domain
C's center of mass, and measure the angle between that vector now vs. at
the reference frame:
    V(t)   = COM(Domain C, aligned frame t) - COM(Domain B, aligned frame t)
    V(ref) = COM(Domain C, reference)       - COM(Domain B, reference)
    angle  = arccos( V(t) . V(ref) / (|V(t)| |V(ref)|) )
This is noisier and less rigorous (no second Kabsch fit to isolate Domain
C's own internal shape change, just a straight-line COM-to-COM direction),
but it's the simpler version the literature itself describes, so worth
having both.

Usage:
    python compute_domain_rotation.py <system> <rep>
    system : with_inhibitor | no_inhibitor
    rep    : 0-4
"""
import sys
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix

system = sys.argv[1].lower()
rep    = int(sys.argv[2])

from config import BASE  # set ALLOSTERY_DATA to override
DYN    = f"{BASE}/dynamics_restrained_malate"
FIXED  = f"{DYN}/pbc_fixed"
ALLOST = f"{BASE}/allostery"
OUT    = f"{ALLOST}/openclosed_metrics"
os.makedirs(OUT, exist_ok=True)

if system == 'with_inhibitor':
    PREFIX = 'me1_239_malate_restrained'
    REF_CRD, REF_PSF = f"{BASE}/me1_239_malate_nad.crd", f"{BASE}/me1_239_malate_nad.psf"
elif system == 'no_inhibitor':
    PREFIX = 'me1_malate_noinhibit_restrained'
    REF_CRD, REF_PSF = f"{BASE}/me1_239_malate_nad_noinhibit.crd", f"{BASE}/me1_239_malate_nad_noinhibit.psf"
else:
    sys.exit(f"Unknown system '{system}' -- expected with_inhibitor or no_inhibitor")

PSF = f"{FIXED}/{PREFIX}_complex.psf"
SEGMENTS = [(i * 100, (i + 1) * 100) for i in range(10)]
OUT_CSV = f"{OUT}/{PREFIX}_{rep}_domain_rotation.csv"

if os.path.isfile(OUT_CSV):
    print(f"EXISTS, skipping: {OUT_CSV}")
    sys.exit(0)

print(f"System: {system}  Replicate: {rep}", flush=True)

DOMAIN_A_SEL = 'resid 17:130'
DOMAIN_B_SEL = 'resid 131:277 or resid 464:535'
DOMAIN_C_SEL = 'resid 278:463'

u_ref = mda.Universe(REF_PSF, REF_CRD)
ab_ref_sel = u_ref.select_atoms(f'segid PROA and name CA and ({DOMAIN_A_SEL} or {DOMAIN_B_SEL})')
b_ref_sel  = u_ref.select_atoms(f'segid PROA and name CA and ({DOMAIN_B_SEL})')
c_ref_sel  = u_ref.select_atoms(f'segid PROA and name CA and ({DOMAIN_C_SEL})')
print(f"Domain A+B CA atoms: {len(ab_ref_sel)}  Domain B-only CA atoms: {len(b_ref_sel)}  "
      f"Domain C CA atoms: {len(c_ref_sel)}", flush=True)

ab_ref_pos = ab_ref_sel.positions.copy()
b_ref_pos = b_ref_sel.positions.copy()
c_ref_pos = c_ref_sel.positions.copy()
ab_ref_com = ab_ref_pos.mean(axis=0)
ab_ref_centered = ab_ref_pos - ab_ref_com
b_ref_centered = b_ref_pos - ab_ref_com   # same frame of reference as A+B
c_ref_centered = c_ref_pos - ab_ref_com

# Reference B-COM -> C-COM vector, for the simpler COM-vector metric
v_ref = c_ref_centered.mean(axis=0) - b_ref_centered.mean(axis=0)
v_ref_norm = v_ref / np.linalg.norm(v_ref)

u = mda.Universe(PSF)
ab_resids = list(ab_ref_sel.resids)
b_resids = list(b_ref_sel.resids)
c_resids = list(c_ref_sel.resids)
ab_idx = np.array([u.select_atoms(f'segid PROA and resid {r} and name CA').indices[0] for r in ab_resids])
b_idx = np.array([u.select_atoms(f'segid PROA and resid {r} and name CA').indices[0] for r in b_resids])
c_idx = np.array([u.select_atoms(f'segid PROA and resid {r} and name CA').indices[0] for r in c_resids])
assert len(ab_idx) == len(ab_ref_pos) and len(c_idx) == len(c_ref_pos)

rows = []
for (start, end) in SEGMENTS:
    dcd = f"{FIXED}/{PREFIX}_{rep}_{start}to{end}ns_fixed.dcd"
    if not os.path.isfile(dcd):
        print(f"  SKIP (missing): {dcd}", flush=True)
        continue

    print(f"Processing {start}-{end} ns ...", flush=True)
    u.load_new(dcd)
    n_frames = len(u.trajectory)

    for iframe, ts in enumerate(u.trajectory):
        t_ns = start + iframe / 100.0

        ab_now = u.atoms.positions[ab_idx]
        b_now = u.atoms.positions[b_idx]
        c_now = u.atoms.positions[c_idx]
        ab_now_com = ab_now.mean(axis=0)
        ab_now_centered = ab_now - ab_now_com
        b_now_centered = b_now - ab_now_com
        c_now_centered = c_now - ab_now_com

        # Step 1-2: superpose A+B onto reference, apply same rotation to
        # Domain C (for the rigorous metric) and to Domain B (for the
        # simpler COM-vector metric) -- same R_ab used for both.
        R_ab, _ = rotation_matrix(ab_now_centered, ab_ref_centered)
        c_in_ab_frame = c_now_centered @ R_ab.T
        b_in_ab_frame = b_now_centered @ R_ab.T

        # Step 3: how much has (now A+B-aligned) Domain C rotated relative
        # to its reference pose?
        R_c, _ = rotation_matrix(c_in_ab_frame, c_ref_centered)

        # Step 4: rotation angle from the rotation matrix trace
        cos_angle = np.clip((np.trace(R_c) - 1) / 2, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Simpler metric: angle between the (aligned) current B->C
        # center-of-mass vector and the reference B->C vector. This one is
        # a literal cosine-similarity angle between two vectors.
        v_now = c_in_ab_frame.mean(axis=0) - b_in_ab_frame.mean(axis=0)
        v_now_dist = np.linalg.norm(v_now)   # Domain B<->C COM distance --
        # validated on static open/closed reference structures
        # (static_structure_metrics.py): 1PJ3 (closed) = 25.13A, clearly
        # shorter than all 5 open references (25.6-26.7A). BSA did NOT
        # validate cleanly there and was dropped; this one did.
        v_now_norm = v_now / v_now_dist
        cos_bc = np.clip(np.dot(v_now_norm, v_ref_norm), -1.0, 1.0)
        bc_angle_deg = np.degrees(np.arccos(cos_bc))

        rows.append({'time_ns': t_ns, 'domain_c_rotation_deg': angle_deg,
                      'domain_bc_com_vector_angle_deg': bc_angle_deg,
                      'domain_bc_com_dist': v_now_dist})

        if (iframe + 1) % 5000 == 0:
            print(f"  Frame {iframe + 1}/{n_frames}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} frames -> {OUT_CSV}", flush=True)
