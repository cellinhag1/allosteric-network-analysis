#!/usr/bin/env python
"""
Compute per-replica DCCM and Gaussian MI (r_MI) matrices from extracted Cα
coordinates, then average over all available replicas.

Gaussian MI follows Lange & Grubmüller, JCP 2006:
  MI_ij  = 0.5 * (log|sigma_i| + log|sigma_j| - log|sigma_ij|)
  r_MI   = sqrt(1 - exp(-2 * MI / 3))   normalised to [0,1]

Usage:
    python compute_dccm_mi.py <condition> [start_frame]

    condition  : me1_inh | me1_noinh | me3_inh | me3_noinh
    start_frame: frame index to start analysis from (default 0 = all 50 ns;
                 use 5000 for last 25 ns only)

Output files in data/:
    <condition>_<tag>_dccm_mean.npy   shape (N, N)   - tag = 'all' or 'last25ns'
    <condition>_<tag>_dccm_std.npy
    <condition>_<tag>_dccm_all.npy    shape (n_reps, N, N) for convergence analysis
    <condition>_<tag>_mi_mean.npy
    <condition>_<tag>_mi_std.npy
    <condition>_<tag>_mi_all.npy
"""
import sys
import os
import numpy as np

condition   = sys.argv[1].lower()
start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0

from config import BASE  # set ALLOSTERY_DATA to override
ALLOD = f'{BASE}/allostery'
DATA  = f'{ALLOD}/data'
N_REPS = 25

tag = 'all' if start_frame == 0 else f'f{start_frame}'
if start_frame == 5000:
    tag = 'last25ns'

print(f"Condition : {condition}", flush=True)
print(f"Frames    : {start_frame} -> end  (tag={tag})", flush=True)


# ── Core computations ─────────────────────────────────────────────────────────

def build_covariance(coords):
    """Full 3N×3N covariance from aligned (n_frames, N, 3) array."""
    n_frames, N, _ = coords.shape
    X  = coords.reshape(n_frames, N * 3).astype(np.float64)
    X -= X.mean(0)
    C  = (X.T @ X) / n_frames
    return C, N


def compute_dccm(C, N):
    """DCCM (N×N) in [-1,1] from 3N×3N covariance matrix."""
    C_r  = C.reshape(N, 3, N, 3)
    # numerator[i,j] = tr(C_ij) = sum_a C[3i+a, 3j+a]
    num  = np.einsum('iaja->ij', C_r)
    var  = np.diag(num)
    denom = np.sqrt(np.maximum(var[:, None] * var[None, :], 1e-30))
    return (num / denom).astype(np.float32)


def compute_mi(C, N):
    """Gaussian r_MI matrix (N×N) in [0,1] from 3N×3N covariance matrix."""
    # 3×3 diagonal blocks
    sigma = np.array([C[3*i:3*i+3, 3*i:3*i+3] for i in range(N)])  # (N,3,3)
    _, log_det_i = np.linalg.slogdet(sigma)                          # (N,)

    # Cross blocks: cross[i,j] = C[3i:3i+3, 3j:3j+3]
    cross = C.reshape(N, 3, N, 3).transpose(0, 2, 1, 3)  # (N,N,3,3)

    # Build (N,N,6,6) joint covariance blocks
    C6 = np.zeros((N, N, 6, 6), dtype=np.float64)
    C6[:, :, :3, :3] = sigma[:, np.newaxis]           # sigma_i block
    C6[:, :, 3:, 3:] = sigma[np.newaxis, :]           # sigma_j block
    C6[:, :, :3, 3:] = cross                           # C_ij
    C6[:, :, 3:, :3] = cross.transpose(1, 0, 3, 2)    # C_ji

    _, log_det_ij = np.linalg.slogdet(C6)              # (N,N)

    MI   = 0.5 * (log_det_i[:, None] + log_det_i[None, :] - log_det_ij)
    MI   = np.maximum(0.0, MI)
    r_MI = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-2.0 * MI / 3.0)))
    return r_MI.astype(np.float32)


# ── Main loop over replicas ───────────────────────────────────────────────────

dccm_list, mi_list, rmsf_list = [], [], []

for rep in range(N_REPS):
    npy = f"{DATA}/{condition}_rep{rep}_ca.npy"
    if not os.path.isfile(npy):
        print(f"  SKIP rep {rep} (no .npy)", flush=True)
        continue

    coords = np.load(npy)                       # (n_kept, N_ca, 3)

    # Apply start_frame filter by *original trajectory frame index* so that
    # the last-25ns window is consistent even when some frames were dropped
    # by the inhibitor distance filter during extraction.
    if start_frame > 0:
        idx_file = f"{DATA}/{condition}_rep{rep}_frame_idx.npy"
        if os.path.isfile(idx_file):
            frame_idx = np.load(idx_file)       # original frame numbers
            mask      = frame_idx >= start_frame
            coords    = coords[mask]
            print(f"  Rep {rep}: {mask.sum()}/{len(mask)} frames "
                  f"with orig_idx >= {start_frame}", flush=True)
        else:
            # No index file → all frames kept; slice by position
            coords = coords[start_frame:]

    n_used = len(coords)
    if n_used < 100:
        print(f"  SKIP rep {rep}: only {n_used} frames after filtering", flush=True)
        continue

    print(f"  Rep {rep}: {n_used} frames", flush=True)
    C, N   = build_covariance(coords)
    dccm   = compute_dccm(C, N)
    mi     = compute_mi(C, N)
    dccm_list.append(dccm)
    mi_list.append(mi)

    # Per-residue RMSF: sqrt(mean(|Δr|²)) over frames
    disp_f64 = coords.astype(np.float64) - coords.mean(0)
    rmsf_rep = np.sqrt((disp_f64 ** 2).sum(-1).mean(0)).astype(np.float32)
    rmsf_list.append(rmsf_rep)

    print(f"    DCCM [{dccm.min():.3f}, {dccm.max():.3f}]  "
          f"MI [{mi.min():.3f}, {mi.max():.3f}]  "
          f"RMSF mean={rmsf_rep.mean():.2f} Å", flush=True)

n_ok = len(dccm_list)
print(f"\n{n_ok}/{N_REPS} replicas processed.", flush=True)

if n_ok == 0:
    print("ERROR: no data found. Run extract_ca_coords.py first.", flush=True)
    sys.exit(1)

dccm_arr = np.stack(dccm_list)
mi_arr   = np.stack(mi_list)
rmsf_arr = np.stack(rmsf_list)

prefix = f"{DATA}/{condition}_{tag}"
np.save(f"{prefix}_dccm_all.npy",  dccm_arr)
np.save(f"{prefix}_dccm_mean.npy", dccm_arr.mean(0))
np.save(f"{prefix}_dccm_std.npy",  dccm_arr.std(0))
np.save(f"{prefix}_mi_all.npy",    mi_arr)
np.save(f"{prefix}_mi_mean.npy",   mi_arr.mean(0))
np.save(f"{prefix}_mi_std.npy",    mi_arr.std(0))
np.save(f"{prefix}_rmsf_mean.npy", rmsf_arr.mean(0))
np.save(f"{prefix}_rmsf_std.npy",  rmsf_arr.std(0))
print(f"Saved -> {prefix}_{{dccm,mi,rmsf}}_{{mean,std[,all]}}.npy", flush=True)
