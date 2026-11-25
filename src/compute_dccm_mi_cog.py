#!/usr/bin/env python
"""
Compute per-replica DCCM, Gaussian MI (r_MI), and RMSF from residue COG
coordinate files (50 ns, up to 25 replicates), then average over replicas.

Reads from data_cog/; output also in data_cog/.

Usage:
    python compute_dccm_mi_cog.py <condition> [start_frame]

    condition  : me1_inh | me1_noinh | me3_inh | me3_noinh
    start_frame: 0 (all 50 ns) or 5000 (last 25 ns)
"""
import sys
import os
import numpy as np

condition   = sys.argv[1].lower()
start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0

from config import BASE  # set ALLOSTERY_DATA to override
ALLOD = f'{BASE}/allostery'
DATA  = f'{ALLOD}/data_cog'
N_REPS = 25

tag = 'all' if start_frame == 0 else f'f{start_frame}'
if start_frame == 5000:
    tag = 'last25ns'

print(f"Condition : {condition}", flush=True)
print(f"Frames    : {start_frame} -> end  (tag={tag})", flush=True)


def build_covariance(coords):
    n_frames, N, _ = coords.shape
    X  = coords.reshape(n_frames, N * 3).astype(np.float64)
    X -= X.mean(0)
    C  = (X.T @ X) / n_frames
    return C, N


def compute_dccm(C, N):
    C_r   = C.reshape(N, 3, N, 3)
    num   = np.einsum('iaja->ij', C_r)
    var   = np.diag(num)
    denom = np.sqrt(np.maximum(var[:, None] * var[None, :], 1e-30))
    return (num / denom).astype(np.float32)


def compute_mi(C, N):
    sigma = np.array([C[3*i:3*i+3, 3*i:3*i+3] for i in range(N)])
    _, log_det_i = np.linalg.slogdet(sigma)

    cross = C.reshape(N, 3, N, 3).transpose(0, 2, 1, 3)

    C6 = np.zeros((N, N, 6, 6), dtype=np.float64)
    C6[:, :, :3, :3] = sigma[:, np.newaxis]
    C6[:, :, 3:, 3:] = sigma[np.newaxis, :]
    C6[:, :, :3, 3:] = cross
    C6[:, :, 3:, :3] = cross.transpose(1, 0, 3, 2)

    _, log_det_ij = np.linalg.slogdet(C6)

    MI   = 0.5 * (log_det_i[:, None] + log_det_i[None, :] - log_det_ij)
    MI   = np.maximum(0.0, MI)
    r_MI = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-2.0 * MI / 3.0)))
    return r_MI.astype(np.float32)


dccm_list, mi_list, rmsf_list = [], [], []

for rep in range(N_REPS):
    npy = f"{DATA}/{condition}_rep{rep}_cog.npy"
    if not os.path.isfile(npy):
        print(f"  SKIP rep {rep} (no .npy)", flush=True)
        continue

    coords = np.load(npy)

    if start_frame > 0:
        idx_file = f"{DATA}/{condition}_rep{rep}_frame_idx.npy"
        if os.path.isfile(idx_file):
            frame_idx = np.load(idx_file)
            mask      = frame_idx >= start_frame
            coords    = coords[mask]
            print(f"  Rep {rep}: {mask.sum()}/{len(mask)} frames "
                  f"with orig_idx >= {start_frame}", flush=True)
        else:
            coords = coords[start_frame:]

    n_used = len(coords)
    if n_used < 100:
        print(f"  SKIP rep {rep}: only {n_used} frames", flush=True)
        continue

    print(f"  Rep {rep}: {n_used} frames", flush=True)
    C, N  = build_covariance(coords)
    dccm  = compute_dccm(C, N)
    mi    = compute_mi(C, N)
    dccm_list.append(dccm)
    mi_list.append(mi)

    disp_f64 = coords.astype(np.float64) - coords.mean(0)
    rmsf_rep = np.sqrt((disp_f64 ** 2).sum(-1).mean(0)).astype(np.float32)
    rmsf_list.append(rmsf_rep)

    print(f"    DCCM [{dccm.min():.3f}, {dccm.max():.3f}]  "
          f"MI [{mi.min():.3f}, {mi.max():.3f}]  "
          f"RMSF mean={rmsf_rep.mean():.2f} A", flush=True)

n_ok = len(dccm_list)
print(f"\n{n_ok}/{N_REPS} replicas processed.", flush=True)

if n_ok == 0:
    print("ERROR: no data found. Run extract_cog_coords.py first.", flush=True)
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
