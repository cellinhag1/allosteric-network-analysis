#!/usr/bin/env python
"""
Strip water from each 100ns DCD segment using MDAnalysis.

Replaces the CHARMM MERGE step in fix_imaging_inhibitor.py, which cannot
handle DCDs with NSAVC=5000 (NFILES header mismatch).

Run with c48a1alf python BEFORE fix_imaging_inhibitor.py:
    python strip_water_mda.py <system> <rep>
"""

import sys
import os
import MDAnalysis as mda

system = sys.argv[1].lower()
rep    = int(sys.argv[2])

from config import BASE  # set ALLOSTERY_DATA to override
DYN  = f"{BASE}/dynamics"
OUT  = f"{DYN}/fixed_dcd"
os.makedirs(OUT, exist_ok=True)

if system == 'me1':
    PSF        = f"{BASE}/me1_239_malate.psf"
    DCD_PREFIX = 'me1_239_malate'
else:
    PSF        = f"{BASE}/me3_239_malate.psf"
    DCD_PREFIX = 'me3_239_malate'

WATER_SEGS = ['WT00', 'WT01', 'WT02', 'WT03']
SEGMENTS   = [(i * 100, (i + 1) * 100) for i in range(10)]

print(f"System: {system.upper()}  Replicate: {rep}", flush=True)

u = mda.Universe(PSF)
complex_sel = u.select_atoms(
    'not (segid ' + ' or segid '.join(WATER_SEGS) + ')'
)
print(f"Complex atoms (non-water): {len(complex_sel)}", flush=True)

for (start, end) in SEGMENTS:
    in_dcd  = f"{DYN}/{DCD_PREFIX}_{rep}_{start}to{end}ns.dcd"
    tmp_dcd = f"{OUT}/{DCD_PREFIX}_{rep}_{start}to{end}ns_tmp.dcd"

    if not os.path.isfile(in_dcd):
        print(f"  SKIP (not found): {in_dcd}", flush=True)
        continue
    if os.path.isfile(tmp_dcd):
        print(f"  EXISTS: {tmp_dcd}", flush=True)
        continue

    print(f"  Stripping {start}-{end}ns ...", flush=True)

    u.load_new(in_dcd)
    n_frames = len(u.trajectory)
    print(f"    Input frames: {n_frames}", flush=True)

    with mda.Writer(tmp_dcd, n_atoms=len(complex_sel)) as W:
        for ts in u.trajectory:
            W.write(complex_sel)

    print(f"    -> {tmp_dcd}  ({n_frames} frames written)", flush=True)

print("Water stripping complete.", flush=True)
