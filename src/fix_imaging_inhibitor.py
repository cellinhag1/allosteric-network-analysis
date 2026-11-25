#!/usr/bin/env python
"""
Fix periodic boundary imaging for ME1/ME3 + LIGAND inhibitor trajectories.

Mirrors the imaging approach from ExtractFrames_cb3_no_cit.ipynb:
  - Phase 1: water stripping done by strip_water_mda.py (MDAnalysis) beforehand
  - Phase 2: Loop frames, centre protein, apply minimum-image convention to
             each ligand (inhibitor, malate, cofactor), write fixed DCD
  - Saves COM distance (LIGAND vs allosteric pocket CA atoms) per frame as .npy

Usage (via SLURM array):
    python fix_imaging_inhibitor.py <system> <rep>
    system : me1 or me3
    rep    : 0-4
"""

import sys
import os
import numpy as np

import pycharmm
from pycharmm import *
import pycharmm.lingo   as lingo
import pycharmm.script  as script
import pycharmm.settings as settings
import pycharmm.read    as read
import pycharmm.write   as write
import pycharmm.coor    as coor
import pycharmm.crystal as crystal
import pycharmm.image   as image
import pycharmm.psf     as psf
import pycharmm.scalar  as scalar

system = sys.argv[1].lower()   # 'me1' or 'me3'
rep    = int(sys.argv[2])       # 0-4

# ── Paths ─────────────────────────────────────────────────────────────────────
from config import BASE  # set ALLOSTERY_DATA to override
DYN        = f"{BASE}/dynamics"
OUT        = f"{DYN}/fixed_dcd"
from config import TOPPAR as TOPPAR_NEW
from config import TOPPAR
from config import CGENFF

os.makedirs(OUT, exist_ok=True)

# ── System-specific configuration ─────────────────────────────────────────────
if system == 'me1':
    PSF          = f"{BASE}/me1_239_malate.psf"
    CRD          = f"{BASE}/me1_239_malate.crd"
    INHIB_SEG    = 'LG00'
    DCD_PREFIX   = 'me1_239_malate'
    POCKET_RESID = [260, 263, 321, 360, 485]   # allosteric pocket CA resids
    STREAM_COF   = os.path.join(CGENFF, "ndp.str")
    STREAM_INHIB = f'{CGENFF}/ligand.str'
    STREAM_MAL   = f'{BASE}/me1_malate.str'
else:  # me3
    PSF          = f"{BASE}/me3_239_malate.psf"
    CRD          = f"{BASE}/me3_239_malate.crd"
    INHIB_SEG    = 'LIGA'
    DCD_PREFIX   = 'me3_239_malate'
    POCKET_RESID = [285, 288, 346, 385, 510]   # allosteric pocket CA resids (ME1+25)
    STREAM_COF   = os.path.join(BASE, "han_cgenff/nap.str")
    STREAM_INHIB = os.path.join(CGENFF, "ligand.str")
    STREAM_MAL   = f'{BASE}/me3_malate.str'

WATER_SEGS = {'WT00', 'WT01', 'WT02', 'WT03'}
COFAC_SEG  = 'LG01'
MALATE_SEG = 'LG03'
SEGMENTS   = [(i * 100, (i + 1) * 100) for i in range(10)]   # 10 × 100 ns

# Per-replicate output paths (avoids race conditions in array jobs)
COMPLEX_PSF = f"{OUT}/{DCD_PREFIX}_rep{rep}_complex.psf"
COMPLEX_PDB = f"{OUT}/{DCD_PREFIX}_rep{rep}_complex_ref.pdb"
COM_NPY     = f"{OUT}/{DCD_PREFIX}_rep{rep}_com_dist.npy"

print(f"System: {system.upper()}  Replicate: {rep}", flush=True)
print(f"Pocket residues: {POCKET_RESID}", flush=True)

# ── Force field ───────────────────────────────────────────────────────────────
settings.set_bomb_level(-2)
read.rtf(f'"{TOPPAR_NEW}/top_all36_prot.rtf"')
read.rtf(f'"{TOPPAR_NEW}/top_all36_cgenff.rtf"', append=True)
read.prm(f'"{TOPPAR_NEW}/par_all36m_prot.prm"', flex=True)
read.prm(f'"{TOPPAR_NEW}/par_all36_cgenff.prm"', flex=True, append=True)
read.stream(f'"{TOPPAR}/toppar_water_ions.str"')
read.stream(f'"{STREAM_COF}"')
read.stream(f'"{STREAM_INHIB}"')
read.stream(f'"{STREAM_MAL}"')
settings.set_bomb_level(0)

# ── Full PSF + CRD ────────────────────────────────────────────────────────────
read.psf_card(f'"{PSF}"')
read.coor_card(f'"{CRD}"')

stats   = coor.stat()
boxsize = max(stats['xmax'] - stats['xmin'],
              stats['ymax'] - stats['ymin'],
              stats['zmax'] - stats['zmin'])
print(f"Initial boxsize: {boxsize:.3f} Ang", flush=True)

# ── Crystal on full system ────────────────────────────────────────────────────
crystal.define_cubic(boxsize)
crystal.build(15.0)


def setup_imaging_full():
    """Image transforms for the full (water-included) system.
       Returns SelectAtoms for the complex (non-water) subset."""
    complex_sel = SelectAtoms()
    for segid in psf.get_segid():
        if segid in WATER_SEGS:
            image.setup_residue(0, 0, 0, 'TIP3')
        elif segid in {'CLA', 'IO00', 'IO01', 'I0O1'}:
            resnms = set(SelectAtoms(seg_id=segid).get_res_names())
            for rn in resnms:
                image.setup_residue(0, 0, 0, rn)
            complex_sel |= SelectAtoms(seg_id=segid)
        else:
            image.setup_segment(0, 0, 0, segid)
            complex_sel |= SelectAtoms(seg_id=segid)
    return complex_sel


complex_full = setup_imaging_full()
n_complex    = sum(list(complex_full))
print(f"Complex (non-water) atoms: {n_complex}", flush=True)

# ── Write complex PSF and reference PDB (used in Phase 2) ─────────────────────
if not os.path.isfile(COMPLEX_PSF) or not os.path.isfile(COMPLEX_PDB):
    write.coor_pdb(f'"{COMPLEX_PDB}"', selection=complex_full)
    crystal.free()
    psf.delete_atoms(~complex_full)
    write.psf_card(f'"{COMPLEX_PSF}"')
    psf.delete_atoms(SelectAtoms(select_all=True))
    print(f"Wrote complex PSF: {COMPLEX_PSF}", flush=True)
    print(f"Wrote complex PDB: {COMPLEX_PDB}", flush=True)
else:
    print(f"EXISTS complex PSF/PDB — skipping write", flush=True)
    crystal.free()
    psf.delete_atoms(SelectAtoms(select_all=True))

# Phase 1 (water stripping) is handled by strip_water_mda.py (MDAnalysis).
# Run that script with c48a1alf python before this one.

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — imaging fix on water-stripped trajectories
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== PHASE 2: Imaging fix ===", flush=True)

read.psf_card(f'"{COMPLEX_PSF}"')
read.pdb(f'"{COMPLEX_PDB}"', resid=True)
crystal.define_cubic(boxsize)
crystal.build(15.0)


def setup_imaging_complex():
    """Image transforms for the complex-only system."""
    complex_c = SelectAtoms()
    for segid in psf.get_segid():
        if segid in {'CLA', 'IO00', 'IO01', 'I0O1'}:
            resnms = set(SelectAtoms(seg_id=segid).get_res_names())
            for rn in resnms:
                image.setup_residue(0, 0, 0, rn)
            complex_c |= SelectAtoms(seg_id=segid)
        else:
            image.setup_segment(0, 0, 0, segid)
            complex_c |= SelectAtoms(seg_id=segid)
    return complex_c


setup_imaging_complex()

# ── Atom index selections ──────────────────────────────────────────────────────
proa_indexes   = SelectAtoms(seg_id='PROA').get_atom_indexes()
inhib_indexes  = SelectAtoms(seg_id=INHIB_SEG).get_atom_indexes()
malate_indexes = SelectAtoms(seg_id=MALATE_SEG).get_atom_indexes()
nadp_indexes   = SelectAtoms(seg_id=COFAC_SEG).get_atom_indexes()

print(f"PROA atoms:  {len(proa_indexes)}", flush=True)
print(f"Inhib atoms: {len(inhib_indexes)} (seg {INHIB_SEG})", flush=True)

# Pocket CA atoms for COM distance — parse complex PDB directly.
# Avoids 'scalar wmain' which is unrecognized in newer CHARMM versions.
_pocket_set = set(POCKET_RESID)
pocket_indexes = []
_atom_idx = 0
with open(COMPLEX_PDB) as _pdb:
    for _line in _pdb:
        if _line.startswith(('ATOM', 'HETATM')):
            _name  = _line[12:16].strip()
            _resid = int(_line[22:26].strip())
            _segid = _line[72:76].strip() if len(_line) > 72 else ''
            if _segid == 'PROA' and _name == 'CA' and _resid in _pocket_set:
                pocket_indexes.append(_atom_idx)
            _atom_idx += 1
print(f"Pocket CA atoms: {len(pocket_indexes)}", flush=True)

# Protein atoms near each ligand (initial reference frame) — used as
# the 'complex centre' for minimum image convention (see getDist)
def get_around_indexes(lig_seg, protein_seg='PROA', cutoff=5.0):
    lig_sel  = SelectAtoms(seg_id=lig_seg)
    prot_sel = SelectAtoms(seg_id=protein_seg)
    around   = ((~lig_sel) & prot_sel) & lig_sel.around(cutoff)
    idx      = around.get_atom_indexes()
    if len(idx) == 0:
        print(f"  WARNING: no protein atoms within {cutoff}Å of {lig_seg}; "
              f"using full PROA as reference.", flush=True)
        idx = proa_indexes
    return idx


around_inhib  = get_around_indexes(INHIB_SEG)
around_malate = get_around_indexes(MALATE_SEG)
around_nadp   = get_around_indexes(COFAC_SEG)

print(f"Reference atoms for MIC: inhib={len(around_inhib)}, "
      f"malate={len(around_malate)}, nadp={len(around_nadp)}", flush=True)


# ── Minimum image correction (from ExtractFrames_cb3_no_cit.ipynb) ────────────
def getDist(lig_indexes, complex_indexes, box):
    """Wrap lig_indexes to nearest image of complex_indexes. Returns (before, after)."""
    xyz      = coor.get_positions()
    c_complex = xyz.iloc[complex_indexes].mean(axis=0)
    c_lig     = xyz.iloc[lig_indexes].mean(axis=0)
    xyz.iloc[lig_indexes] -= np.round((c_lig / box)) * box
    before = np.linalg.norm(c_complex - c_lig)
    xyz.iloc[lig_indexes] += np.round(((c_complex - c_lig) / box)) * box
    after  = np.linalg.norm(c_complex - xyz.iloc[lig_indexes].mean(axis=0))
    coor.set_positions(xyz)
    return (before, after)


# ── Per-segment imaging loop ───────────────────────────────────────────────────
all_com_dists = []

for (start, end) in SEGMENTS:
    tmp_dcd = f"{OUT}/{DCD_PREFIX}_{rep}_{start}to{end}ns_tmp.dcd"
    out_dcd = f"{OUT}/{DCD_PREFIX}_{rep}_{start}to{end}ns_fixed.dcd"

    if not os.path.isfile(tmp_dcd):
        print(f"  SKIP (no tmp DCD): {tmp_dcd}", flush=True)
        continue
    if os.path.isfile(out_dcd):
        print(f"  EXISTS: {out_dcd}  — loading saved COM for this segment", flush=True)
        # Load partial data if already done
        partial_npy = out_dcd.replace('_fixed.dcd', '_com_dist.npy')
        if os.path.isfile(partial_npy):
            all_com_dists.extend(np.load(partial_npy).tolist())
        os.remove(tmp_dcd) if os.path.isfile(tmp_dcd) else None
        continue

    print(f"\nProcessing {start}-{end} ns ...", flush=True)

    dcd_in_f  = CharmmFile(f'"{tmp_dcd}"', read_only=True,  formatted=False)
    dcd_out_f = CharmmFile(f'"{out_dcd}"', read_only=False, formatted=False)

    script.CommandScript('traj', query=True, unit=dcd_in_f.file_unit).run()
    nframes = int(lingo.get_energy_value('NFILE'))
    print(f"  Frames in tmp DCD: {nframes}", flush=True)

    script.CommandScript(
        'traj', nunit=1,
        firstunit=dcd_in_f.file_unit,
        nwrite=1,
        iwrite=dcd_out_f.file_unit
    ).run()

    seg_com = []

    for iframe in range(nframes):
        script.CommandScript('traj', read=True).run()

        # Per-frame box dimensions from DCD header
        abc = np.array([lingo.get_energy_value('XTLA'),
                        lingo.get_energy_value('XTLB'),
                        lingo.get_energy_value('XTLC')])
        # MDAnalysis-written DCDs may not carry crystal info; fall back to initial boxsize
        if abc[0] < 1.0:
            abc = np.array([boxsize, boxsize, boxsize])

        # 1. Centre protein in box
        curr_xyz = coor.get_positions()
        curr_xyz.iloc[proa_indexes] -= (
            np.round(curr_xyz.iloc[proa_indexes].mean(axis=0) / abc) * abc
        )
        coor.set_positions(curr_xyz)

        # 2. Minimum image for each ligand relative to its protein neighbourhood
        getDist(inhib_indexes,  around_inhib,  abc)
        getDist(malate_indexes, around_malate, abc)
        getDist(nadp_indexes,   around_nadp,   abc)

        # 3. COM distance: LIGAND vs allosteric pocket
        curr_xyz  = coor.get_positions()
        inhib_com  = curr_xyz.iloc[inhib_indexes].mean(axis=0).values
        pocket_com = curr_xyz.iloc[pocket_indexes].mean(axis=0).values
        seg_com.append(float(np.linalg.norm(inhib_com - pocket_com)))

        # 4. Write imaging-corrected frame (complex only, no water)
        script.CommandScript('traj', write=True).run()

        if (iframe + 1) % 2000 == 0:
            print(f"  Frame {iframe + 1}/{nframes}  "
                  f"COM={seg_com[-1]:.2f} Ang", flush=True)

    dcd_in_f.close()
    dcd_out_f.close()

    # Save per-segment COM as checkpoint
    seg_npy = out_dcd.replace('_fixed.dcd', '_com_dist.npy')
    np.save(seg_npy, np.array(seg_com))

    all_com_dists.extend(seg_com)
    print(f"  Done. mean COM = {np.mean(seg_com):.2f} Ang  "
          f"({np.sum(np.array(seg_com) > 10)}/{nframes} frames > 10 Ang)", flush=True)

    os.remove(tmp_dcd)

# ── Save full COM distance time series ────────────────────────────────────────
if all_com_dists:
    com_arr = np.array(all_com_dists)
    np.save(COM_NPY, com_arr)
    print(f"\nSaved COM distances -> {COM_NPY}  ({len(com_arr)} frames)", flush=True)
    print(f"COM distance: mean={com_arr.mean():.2f} "
          f"std={com_arr.std():.2f} "
          f"max={com_arr.max():.2f} Ang", flush=True)

crystal.free()
psf.delete_atoms(SelectAtoms(select_all=True))
print("\nDone.", flush=True)
