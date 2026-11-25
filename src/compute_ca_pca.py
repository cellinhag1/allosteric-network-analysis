#!/usr/bin/env python
"""
PCA (essential dynamics) of per-residue Ca motion, plus an L1-norm
alternative, to identify which residues contribute most to the dominant
collective modes of motion -- with vs. without inhibitor, for both the
main restrained dataset (A) and the second/older unrestrained dataset (B).

Method, per replicate (each already Kabsch-aligned to its OWN first frame
by the extraction scripts -- covariance is computed per-replicate rather
than pooling raw frames across replicates, avoiding spurious variance from
replicates being aligned to different references; this matches how
compute_dccm_mi_cog.py already handles multi-replicate DCCM/MI averaging
in this project):

  1. Build the (3N x 3N) mass-unweighted covariance matrix of Ca
     displacements from that replicate's own mean structure:
         X = coords.reshape(n_frames, N*3);  X -= X.mean(0)
         C = (X.T @ X) / n_frames
  2. Eigendecompose C (symmetric -> np.linalg.eigh): eigenvalues =
     variance explained per mode, eigenvectors = the modes themselves
     (each a 3N-dim unit vector, i.e. a per-residue 3-component
     displacement direction).
  3. Per-residue contribution to the top-K modes, two ways:
       L2 (standard "collectivity"): for residue i, sum over the top K
         modes of eigenvalue_k * norm(eigenvector_k's 3-component slice
         for residue i)^2 -- large motions in high-variance modes count
         more, and outlier residues can dominate.
       L1 (robust alternative): same but with the L1 (absolute value)
         norm of each mode's per-residue slice instead of squared norm --
         less sensitive to a small number of large-amplitude residues.
  4. Average these per-residue scores across replicates (simple mean;
     replicates already carry per-frame filtering/frame-count differences
     internally, e.g. the inhibitor-bound-frame filter for with_inhibitor).

K (number of top modes) is chosen per replicate as however many are needed
to explain TOP_VARIANCE_FRAC of total variance (default 50%), so it
adapts to how concentrated/spread out each replicate's motion actually is,
rather than an arbitrary fixed mode count.

Usage:
    python compute_ca_pca.py <dataset> <condition>
    dataset   : A (data_restrained/, with_inhibitor|no_inhibitor) |
                B (data_long/, me1_inh|me1_noinh)
    condition : with_inhibitor | no_inhibitor   (dataset A)
                me1_inh | me1_noinh              (dataset B)

Output (in pca_results/):
    {dataset}_{condition}_pca.npz with keys:
        resids            (N,)
        per_rep_l2        (n_reps, N)   -- per-replicate L2 scores
        per_rep_l1        (n_reps, N)   -- per-replicate L1 scores
        mean_l2, std_l2   (N,)          -- averaged across replicates
        mean_l1, std_l1   (N,)
        eigenvalue_frac   (n_reps,)     -- variance explained by top-1 mode, per rep (sanity check)
        n_modes_used      (n_reps,)     -- how many modes needed for TOP_VARIANCE_FRAC, per rep
"""
import sys
import os
import numpy as np

TOP_VARIANCE_FRAC = 0.5  # use as many top modes as needed to explain this fraction of total variance

dataset = sys.argv[1].upper()
condition = sys.argv[2].lower()

from config import BASE  # set ALLOSTERY_DATA to override
OUT_DIR = f"{BASE}/pca_results"
os.makedirs(OUT_DIR, exist_ok=True)

if dataset == 'A':
    DATA = f"{BASE}/data_restrained"
    if condition not in ('with_inhibitor', 'no_inhibitor'):
        sys.exit("dataset A conditions: with_inhibitor | no_inhibitor")
    N_REPS = 5
    coord_fmt = f"{DATA}/{condition}_rep{{rep}}_ca.npy"
    resid_file = f"{DATA}/{condition}_resids.npy"
elif dataset == 'B':
    DATA = f"{BASE}/data_long"
    if condition not in ('me1_inh', 'me1_noinh'):
        sys.exit("dataset B conditions: me1_inh | me1_noinh")
    N_REPS = 5
    coord_fmt = f"{DATA}/{condition}_rep{{rep}}_ca.npy"
    resid_file = f"{DATA}/{condition}_resids.npy"
elif dataset == 'C':
    DATA = f"{BASE}/data"
    if condition not in ('me1_inh', 'me1_noinh'):
        sys.exit("dataset C conditions: me1_inh | me1_noinh")
    N_REPS = 25
    coord_fmt = f"{DATA}/{condition}_rep{{rep}}_ca.npy"
    resid_file = f"{DATA}/{condition}_resids.npy"
elif dataset == 'AB':
    # Dataset A's 5 reps + dataset B's 5 reps pooled as "10 replicates" of one
    # combined ensemble -- to see whether adding A's frames changes the PCA
    # picture relative to B alone. Both datasets already share the same
    # Domain-A+B-anchored Kabsch alignment and the same resid numbering, so
    # pooling their per-replicate covariance/score contributions is valid.
    if condition not in ('with_inhibitor', 'no_inhibitor'):
        sys.exit("dataset AB conditions: with_inhibitor | no_inhibitor")
    DATA_A = f"{BASE}/data_restrained"
    DATA_B = f"{BASE}/data_long"
    COND_B = {'with_inhibitor': 'me1_inh', 'no_inhibitor': 'me1_noinh'}[condition]
    N_REPS = 10

    def coord_fmt_ab(rep):
        return (f"{DATA_A}/{condition}_rep{rep}_ca.npy" if rep < 5
                else f"{DATA_B}/{COND_B}_rep{rep - 5}_ca.npy")
    resid_file = f"{DATA_A}/{condition}_resids.npy"
else:
    sys.exit("dataset must be A, B, C, or AB")

resids = np.load(resid_file)
N = len(resids)
print(f"dataset={dataset} condition={condition}  N_residues={N}", flush=True)


N_SCREE = 10  # how many top modes to record for the scree/variance-explained plot


def per_residue_scores(C, N_, top_variance_frac, n_scree):
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]  # descending order, columns = modes
    eigvals = np.clip(eigvals, 0, None)
    total_var = eigvals.sum()
    frac = np.cumsum(eigvals) / total_var
    n_modes = int(np.searchsorted(frac, top_variance_frac) + 1)
    n_modes = max(1, min(n_modes, len(eigvals)))

    top_eigvals = eigvals[:n_modes]
    top_eigvecs = eigvecs[:, :n_modes].reshape(N_, 3, n_modes)  # (N, 3, n_modes)

    l2_score = np.zeros(N_)
    l1_score = np.zeros(N_)
    for k in range(n_modes):
        vec_k = top_eigvecs[:, :, k]  # (N, 3)
        l2_score += top_eigvals[k] * (vec_k ** 2).sum(axis=1)
        l1_score += top_eigvals[k] * np.abs(vec_k).sum(axis=1)

    scree_frac = (eigvals[:n_scree] / total_var)  # variance explained by each of the top n_scree PCs individually
    pc1_pc2_vecs = eigvecs[:, :2]  # (3N, 2) -- for projecting frames onto PC1/PC2

    return l2_score, l1_score, n_modes, eigvals[0] / total_var, scree_frac, pc1_pc2_vecs


per_rep_l2, per_rep_l1, n_modes_used, top1_frac = [], [], [], []
per_rep_scree = []       # (n_reps, N_SCREE) variance-explained-per-PC
per_rep_pc12 = []         # object array: one (n_frames, 2) projection array per replicate

for rep in range(N_REPS):
    npy = coord_fmt_ab(rep) if dataset == 'AB' else coord_fmt.format(rep=rep)
    if not os.path.isfile(npy):
        print(f"  SKIP rep {rep}: file not found ({npy})", flush=True)
        continue
    coords = np.load(npy)
    n_frames = len(coords)
    if n_frames < 100:
        print(f"  SKIP rep {rep}: only {n_frames} frames", flush=True)
        continue

    X = coords.reshape(n_frames, N * 3).astype(np.float64)
    Xc = X - X.mean(0)
    C = (Xc.T @ Xc) / n_frames
    l2, l1, n_modes, top1, scree_frac, pc12_vecs = per_residue_scores(C, N, TOP_VARIANCE_FRAC, N_SCREE)
    per_rep_l2.append(l2)
    per_rep_l1.append(l1)
    n_modes_used.append(n_modes)
    top1_frac.append(top1)
    per_rep_scree.append(scree_frac)
    per_rep_pc12.append(Xc @ pc12_vecs)  # (n_frames, 2) projection of each frame onto PC1/PC2
    print(f"  rep {rep}: {n_frames} frames, {n_modes} modes explain {TOP_VARIANCE_FRAC:.0%} of variance, "
          f"top-1 mode alone = {top1:.1%}", flush=True)

per_rep_l2 = np.array(per_rep_l2)
per_rep_l1 = np.array(per_rep_l1)
per_rep_scree = np.array(per_rep_scree)
per_rep_pc12 = np.array(per_rep_pc12, dtype=object)

mean_l2, std_l2 = per_rep_l2.mean(axis=0), per_rep_l2.std(axis=0)
mean_l1, std_l1 = per_rep_l1.mean(axis=0), per_rep_l1.std(axis=0)

out_npz = f"{OUT_DIR}/{dataset}_{condition}_pca.npz"
np.savez_compressed(
    out_npz,
    resids=resids,
    per_rep_l2=per_rep_l2, per_rep_l1=per_rep_l1,
    mean_l2=mean_l2, std_l2=std_l2,
    mean_l1=mean_l1, std_l1=std_l1,
    eigenvalue_frac=np.array(top1_frac),
    n_modes_used=np.array(n_modes_used),
    per_rep_scree=per_rep_scree,       # (n_reps, N_SCREE) variance explained by each of top N_SCREE PCs
    per_rep_pc12=per_rep_pc12,         # object array of (n_frames, 2) PC1/PC2 projections, one per rep
)
print(f"Wrote {out_npz}", flush=True)

top5_l2 = resids[np.argsort(mean_l2)[::-1][:5]]
top5_l1 = resids[np.argsort(mean_l1)[::-1][:5]]
print(f"Top 5 residues by L2 score: {list(top5_l2)}", flush=True)
print(f"Top 5 residues by L1 score: {list(top5_l1)}", flush=True)
