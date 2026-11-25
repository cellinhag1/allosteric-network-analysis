#!/usr/bin/env python
"""
Network Build — Step 1 of allosteric network analysis.

Usage:
    python network_build.py <system>
    system must be one of: "ME1+Inhib"  "ME3+Inhib"

Only inhibitor-bound frames are used (LIGAND CoM → pocket centroid < cutoff).
Run one job per system so both can run in parallel on the cluster.

Steps per system:
  1. Filter frames: LIGAND CoM → allosteric pocket Cα centroid < 8 Å
  2. Extract protein Cα positions at ~100 ps (segment-by-segment with stride)
  3. Cα–Cα contact persistence (< 7 Å in > 25 % of bound frames)
  4. Generalised correlation r_MI per contact (Gaussian MI approximation,
     Lange & Grubmüller 2006, J Chem Phys 124:214903)
  5. Save weighted NetworkX graphs (per-replicate + mean)

Outputs in BASE/networks/:
  {sys}_rep{n}_graph.pkl    per-replicate graph
  {sys}_mean_graph.pkl      mean graph averaged across 5 replicates
  {sys}_fig_ligand_filter.png  LIGAND distance distribution + bound-frame fraction
  {sys}_fig_rMI_matrix.png  generalised correlation heatmap
"""

import sys, os, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import MDAnalysis as mda

# ── Which system to run (passed as command-line argument) ─────────────────────
VALID = ["ME1+Inhib", "ME1-noInhib", "ME3+Inhib"]
if len(sys.argv) < 2 or sys.argv[1] not in VALID:
    print(f"Usage: python network_build.py <system>")
    print(f"  system must be one of: {VALID}")
    sys.exit(1)
RUN_SYSTEM = sys.argv[1]
print(f"\nRunning system: {RUN_SYSTEM}\n")

from config import BASE  # set ALLOSTERY_DATA to override
DYN  = f"{BASE}/dynamics"
OUT  = f"{BASE}/networks"
os.makedirs(OUT, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
CUTOFF_CA        = 7.0    # Å  Cα–Cα contact distance (PSN standard)
PERSIST_THRESH   = 0.25   # fraction of bound frames required to keep an edge
LIGAND_BOUND_CUTOFF = 8.0    # Å  LIGAND CoM → pocket centroid = "inhibitor bound"
EQUIL_NS         = 10.0   # ns to discard as equilibration (first segment only)
STRIDE_FINE      = 50     # every 50th frame of 2-ps segments  → 100 ps spacing
STRIDE_COARSE    = 10     # every 10th frame of 10-ps segments → 100 ps spacing
N_REPS           = 5

# ── System definitions ────────────────────────────────────────────────────────
SYSTEMS = {
    "ME1+Inhib": {
        "psf":           f"{BASE}/me1_239_malate.psf",
        "dcd_glob":      f"{DYN}/me1_239_malate_{{rep}}_*to*ns.dcd",
        "ligand_sel":       "segid LG00 and resname LIGAND and not name H*",
        "pocket_resids": [249,250,270,271,274,303,304,311,312,350,353],
        "active_resids": [112,165,183,418,463],
        "has_inhibitor": True,
        "color":         "#2166ac",
    },
    "ME1-noInhib": {
        "psf":           f"{BASE}/me1_malate_no_inhibitor.psf",
        "dcd_glob":      f"{DYN}/me1_malate_no_inhibitor{{rep}}_*to*ns.dcd",
        "ligand_sel":       None,    # no inhibitor — all production frames used
        "pocket_resids": [249,250,270,271,274,303,304,311,312,350,353],
        "active_resids": [112,165,183,418,463],
        "has_inhibitor": False,
        "color":         "#4dac26",
    },
    "ME3+Inhib": {
        "psf":           f"{BASE}/me3_239_malate.psf",
        "dcd_glob":      f"{DYN}/me3_239_malate_{{rep}}_*to*ns.dcd",
        "ligand_sel":       "segid LIGA and resname LIGAND and not name H*",
        "pocket_resids": [284,305,306,309,337,338,339,342,346,347,383,385,386],
        "active_resids": [137,190,208,443,488],
        "has_inhibitor": True,
        "color":         "#d6604d",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def sorted_segments(pattern):
    files = glob.glob(pattern)
    # drop zero-byte or missing files — these are placeholder files from
    # dynamics segments that haven't finished writing yet
    files = [f for f in files if os.path.getsize(f) > 0]
    def start_ns(f):
        tag = os.path.basename(f).replace(".dcd","").split("_")[-1]
        return int(tag.split("to")[0])
    return sorted(files, key=start_ns)

def seg_stride(seg_path):
    """Return (stride, is_fine) based on segment name."""
    tag = os.path.basename(seg_path).replace(".dcd","").split("_")[-1]
    start = int(tag.split("to")[0])
    if start < 200:          # 0–100 ns and 100–200 ns are at 2 ps/frame
        return STRIDE_FINE, True
    return STRIDE_COARSE, False

def equil_frame_cutoff(seg_path):
    """Frames to skip at start of segment for equilibration (first seg only)."""
    tag = os.path.basename(seg_path).replace(".dcd","").split("_")[-1]
    if int(tag.split("to")[0]) == 0:
        # first segment: skip first EQUIL_NS nanoseconds at 2 ps/frame
        return int(EQUIL_NS * 1000 / 2)
    return 0

def seg_time_offset_ns(seg_path):
    """Time (ns) at the first frame of this segment."""
    tag = os.path.basename(seg_path).replace(".dcd","").split("_")[-1]
    return float(tag.split("to")[0])

def generalized_correlation(xi, xj):
    """
    Generalised correlation r_MI for two (N,3) Cα displacement time series.
    Gaussian MI approximation — Lange & Grubmüller 2006, J Chem Phys 124:214903.
    r_MI = sqrt(1 - exp(-2*MI/3)),  MI from 6×6 joint covariance (log-det formula).
    Returns float in [0, 1].
    """
    xi = xi - xi.mean(0)
    xj = xj - xj.mean(0)
    xy = np.hstack([xi, xj])
    n  = xy.shape[0]

    cov_i  = (xi.T @ xi) / (n - 1)
    cov_j  = (xj.T @ xj) / (n - 1)
    cov_ij = (xy.T @ xy) / (n - 1)

    _, ld_i  = np.linalg.slogdet(cov_i)
    _, ld_j  = np.linalg.slogdet(cov_j)
    s,  ld_ij = np.linalg.slogdet(cov_ij)

    if s <= 0:
        return 0.0

    MI = max(0.5 * (ld_i + ld_j - ld_ij), 0.0)
    return float(np.sqrt(1.0 - np.exp(-2.0 * MI / 3.0)))

# ── LIGAND filter figure (single system) ────────────────────────────────────────
fig_filt, ax_filt = plt.subplots(figsize=(6, 4))

# ── Main loop (single system only) ───────────────────────────────────────────
all_mean_graphs = {}
cfg = SYSTEMS[RUN_SYSTEM]
sys_name = RUN_SYSTEM

if True:   # indent block preserved — was a for loop
    print(f"\n{'='*65}")
    print(f"  {sys_name}")
    print(f"{'='*65}")

    rep_graphs       = []
    rep_rMI_acc      = {}   # (i,j) → cumulative r_MI across reps
    rep_persist_acc  = {}
    rep_count        = 0

    for rep in range(N_REPS):
        pattern = cfg["dcd_glob"].format(rep=rep)
        segs    = sorted_segments(pattern)
        if not segs:
            print(f"  Rep {rep}: no trajectories found — skipping")
            continue
        print(f"\n  Rep {rep}: {len(segs)} segments")

        # ── Build protein-only universe once from PSF to get residue info ──
        u_ref = mda.Universe(cfg["psf"])
        protein_ref  = u_ref.select_atoms("protein and name CA")
        n_res        = len(protein_ref)
        resids_arr   = np.array([r.resid for r in protein_ref.residues])
        resnames_arr = np.array([r.resname for r in protein_ref.residues])

        pocket_idx = np.array([np.where(resids_arr == r)[0][0]
                                for r in cfg["pocket_resids"] if r in resids_arr])
        active_idx = np.array([np.where(resids_arr == r)[0][0]
                                for r in cfg["active_resids"] if r in resids_arr])

        if rep == 0 and sys_name == list(SYSTEMS.keys())[0]:
            print(f"    Protein Cα count: {n_res}, resid {resids_arr[0]}–{resids_arr[-1]}")
        print(f"    Pocket residues matched: {len(pocket_idx)}/{len(cfg['pocket_resids'])}")
        print(f"    Active residues matched: {len(active_idx)}/{len(cfg['active_resids'])}")

        # ── Iterate segments one at a time to avoid loading full 1 µs at once
        all_ligand_dists = []   # for filter figure
        ca_frames     = []   # list of (n_res,3) arrays — bound frames only

        for seg in segs:
            stride, is_fine = seg_stride(seg)
            skip_start      = equil_frame_cutoff(seg)
            t_offset_ns     = seg_time_offset_ns(seg)
            ps_per_frame    = 2 if is_fine else 10

            # Load segment with protein + inhibitor (waters ignored in computation
            # but still read by MDAnalysis; striding minimises frames read)
            u_seg = mda.Universe(cfg["psf"], seg)

            ca_seg   = u_seg.select_atoms("protein and name CA")
            ligand_seg  = u_seg.select_atoms(cfg["ligand_sel"]) if cfg["has_inhibitor"] else None

            n_frames_seg = len(u_seg.trajectory)
            # frame indices to read for this segment
            read_indices = range(skip_start, n_frames_seg, stride)

            n_bound_seg = 0
            try:
                for local_f in read_indices:
                    ts = u_seg.trajectory[local_f]

                    if cfg["has_inhibitor"]:
                        pocket_centroid = ca_seg[pocket_idx].positions.mean(0)
                        ligand_com         = ligand_seg.center_of_mass()
                        ligand_d           = float(np.linalg.norm(ligand_com - pocket_centroid))
                        all_ligand_dists.append(ligand_d)
                        if ligand_d >= LIGAND_BOUND_CUTOFF:
                            continue   # inhibitor not in pocket — skip frame

                    ca_frames.append(ca_seg.positions.copy())
                    n_bound_seg += 1
            except (OSError, EOFError, Exception) as e:
                print(f"      WARNING: {os.path.basename(seg)} truncated after "
                      f"{n_bound_seg} frames ({e}) — using partial data")

            print(f"      {os.path.basename(seg)}: "
                  f"{len(read_indices)} read, {n_bound_seg} kept")

        # ── Report bound fraction ──────────────────────────────────────────
        n_total = len(ca_frames)
        pct = 100 * n_total / len(all_ligand_dists) if all_ligand_dists else 100.0
        print(f"    Bound frames: {n_total:,} / {len(all_ligand_dists):,}  ({pct:.1f}%)")
        if all_ligand_dists:
            ax_filt.hist(all_ligand_dists, bins=80, alpha=0.45, density=True, label=f"Rep {rep}")

        if n_total < 200:
            print(f"    WARNING: fewer than 200 frames — skipping this replicate")
            continue

        # ── Contact persistence ────────────────────────────────────────────
        print(f"    Computing contact map ({n_total} frames) ...")
        ca_arr = np.array(ca_frames)   # (N, n_res, 3)

        contact_sum = np.zeros((n_res, n_res), dtype=np.float32)
        BATCH = 200
        for b in range(0, n_total, BATCH):
            batch = ca_arr[b:b+BATCH]                                  # (B, R, 3)
            diff  = batch[:, :, None, :] - batch[:, None, :, :]       # (B, R, R, 3)
            d     = np.sqrt((diff**2).sum(-1))                         # (B, R, R)
            contact_sum += (d < CUTOFF_CA).sum(0).astype(np.float32)

        persistence = contact_sum / n_total
        np.fill_diagonal(persistence, 0.0)

        # upper-triangle contact pairs above threshold (exclude neighbours i±1,2)
        mask = (persistence > PERSIST_THRESH) & \
               (np.triu(np.ones((n_res, n_res), dtype=bool), k=3))
        pairs = list(zip(*np.where(mask)))
        print(f"    Persistent contacts: {len(pairs):,}")

        # ── Generalised correlation ────────────────────────────────────────
        print(f"    Computing generalised correlation ...")
        ca_disp = ca_arr - ca_arr.mean(0, keepdims=True)   # (N, n_res, 3)

        G = nx.Graph()
        for i, res in enumerate(protein_ref.residues):
            G.add_node(i,
                       resid    = int(res.resid),
                       resname  = res.resname,
                       is_pocket= int(res.resid) in cfg["pocket_resids"],
                       is_active= int(res.resid) in cfg["active_resids"])

        for (i, j) in pairs:
            r_mi = generalized_correlation(ca_disp[:, i, :], ca_disp[:, j, :])
            if r_mi > 1e-6:
                w = float(-np.log(r_mi + 1e-10))
                G.add_edge(i, j, r_MI=r_mi, weight=w,
                           persistence=float(persistence[i, j]))
                key = (min(i,j), max(i,j))
                rep_rMI_acc[key]     = rep_rMI_acc.get(key, 0.0)     + r_mi
                rep_persist_acc[key] = rep_persist_acc.get(key, 0.0) + float(persistence[i,j])

        print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges")

        sys_tag = sys_name.replace("+","p").replace("-","n")
        pkl = f"{OUT}/{sys_tag}_rep{rep}_graph.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(G, fh)
        print(f"    Saved {pkl}")

        rep_graphs.append(G)
        rep_count += 1

    # ── Mean graph across replicates ─────────────────────────────────────────
    if rep_count == 0:
        print(f"  No valid replicates for {sys_name} — nothing to save")
        sys.exit(1)

    print(f"\n  Building mean graph from {rep_count} replicates ...")
    Gref = rep_graphs[0]
    Gmean = nx.Graph()
    for n, d in Gref.nodes(data=True):
        Gmean.add_node(n, **d)

    for key, rmi_sum in rep_rMI_acc.items():
        i, j   = key
        r_mi   = rmi_sum / rep_count
        p      = rep_persist_acc[key] / rep_count
        Gmean.add_edge(i, j, r_MI=r_mi,
                       weight=float(-np.log(r_mi + 1e-10)),
                       persistence=p)

    sys_tag  = sys_name.replace("+","p").replace("-","n")
    mean_pkl = f"{OUT}/{sys_tag}_mean_graph.pkl"
    with open(mean_pkl, "wb") as fh:
        pickle.dump(Gmean, fh)
    print(f"  Saved mean graph → {mean_pkl}  "
          f"({Gmean.number_of_nodes()} nodes, {Gmean.number_of_edges():,} edges)")
    all_mean_graphs[sys_name] = Gmean

# ── LIGAND filter figure ─────────────────────────────────────────────────────────
tag = RUN_SYSTEM.replace("+","p").replace("-","n")
ax_filt.axvline(LIGAND_BOUND_CUTOFF, color="red", lw=1.5, ls="--")
ax_filt.set_xlabel("LIGAND CoM → pocket centroid (Å)", fontsize=10)
ax_filt.set_ylabel("Density", fontsize=10)
ax_filt.set_title(f"{RUN_SYSTEM} — LIGAND distance distribution", fontsize=11, fontweight="bold")
ax_filt.legend(fontsize=7)
fig_filt.suptitle("Frames left of red line = inhibitor-bound frames used for network",
                  fontsize=10)
fig_filt.tight_layout()
filt_path = f"{OUT}/{tag}_fig_ligand_filter.png"
fig_filt.savefig(filt_path, dpi=200, bbox_inches="tight")
print(f"\nSaved {filt_path}")

# ── r_MI heatmap ──────────────────────────────────────────────────────────────
if RUN_SYSTEM in all_mean_graphs:
    G   = all_mean_graphs[RUN_SYSTEM]
    n   = G.number_of_nodes()
    mat = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        mat[i, j] = mat[j, i] = d["r_MI"]
    fig_h, ax_h = plt.subplots(figsize=(7, 6))
    im = ax_h.imshow(mat, cmap="hot_r", vmin=0, vmax=1, aspect="auto")
    ax_h.set_title(f"{RUN_SYSTEM} — Generalised correlation (mean across replicates)",
                   fontsize=11, fontweight="bold")
    ax_h.set_xlabel("Residue index")
    ax_h.set_ylabel("Residue index")
    plt.colorbar(im, ax=ax_h, label="r_MI")
    fig_h.tight_layout()
    rmi_path = f"{OUT}/{tag}_fig_rMI_matrix.png"
    fig_h.savefig(rmi_path, dpi=200, bbox_inches="tight")
    print(f"Saved {rmi_path}")

print(f"\n{'='*65}")
print(f"All done. Graphs saved to {OUT}/")
