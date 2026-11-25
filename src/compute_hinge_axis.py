#!/usr/bin/env python
"""
Track the hinge ROTATION AXIS over time, not just the rotation ANGLE
(theta). CHARMM's coor.orient() gives us theta (via ['THET']) but not the
axis directly -- so this reconstructs the same Step-2 rotation matrix
ourselves (Domain C fit onto its reference, on top of the already
Domain-A+B-aligned coordinates) and extracts the axis from it directly.

Reuses the already-extracted, already Domain-A+B-aligned Cα coordinates in
data_restrained/ -- these are already "Step 1" of the hinge calculation
(fit on Domain A+B, applied to everything), so we only need to add "Step
2" (fit Domain C specifically) to get the exact same rotation matrix
CHARMM computes for theta -- and this time we keep the whole matrix
instead of just its angle.

For a rotation matrix R with angle theta, the rotation AXIS (unit vector)
comes from R's antisymmetric part:
    axis_raw = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]
    axis = axis_raw / (2*sin(theta))
Sign convention is arbitrary (axis and -axis describe related but
different-handedness rotations) -- we fix this by flipping each frame's
axis to keep it as close as possible to the previous frame's, so a
smoothly-varying true axis doesn't show up as spurious 180-degree flips.

Usage:
    python compute_hinge_axis.py <condition> <rep>   (with_inhibitor | no_inhibitor, 0-4)

Output: pca_results/hinge_axis/A_{condition}_rep{rep}_axis.npz
    theta_deg   (n_frames,)  -- rotation angle, should match CHARMM's THET closely
    axis        (n_frames, 3)  -- unit rotation axis, sign-consistent frame to frame
    wander_deg  (n_frames,)  -- angle between this frame's axis and the trajectory's mean axis
"""
import sys
import numpy as np
import os

condition = sys.argv[1]
rep = int(sys.argv[2])

from config import BASE  # set ALLOSTERY_DATA to override
DATA = f"{BASE}/data_restrained"
OUT_DIR = f"{BASE}/pca_results/hinge_axis"
os.makedirs(OUT_DIR, exist_ok=True)

DOMAIN_C = (278, 463)

resids = np.load(f"{DATA}/{condition}_resids.npy")
c_mask = (resids >= DOMAIN_C[0]) & (resids <= DOMAIN_C[1])
assert c_mask.sum() == 186, f"Expected 186 Domain C atoms, got {c_mask.sum()}"

coords = np.load(f"{DATA}/{condition}_rep{rep}_ca.npy").astype(np.float64)  # (n_frames, 548, 3)
time_ns = np.load(f"{DATA}/{condition}_rep{rep}_time_ns.npy")
n_frames = coords.shape[0]

ref_c = coords[0, c_mask, :]                 # Domain C reference = this replicate's own first frame
ref_c_com = ref_c.mean(0)
ref_c_centered = ref_c - ref_c_com

pos_c = coords[:, c_mask, :]                  # (n_frames, 186, 3)
com_c = pos_c.mean(axis=1, keepdims=True)
pos_c_centered = pos_c - com_c


def kabsch_rot_batch(P, Q):
    H = np.einsum('nka,kb->nab', P, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    diag = np.zeros((len(d), 3, 3))
    diag[:, 0, 0] = 1.0
    diag[:, 1, 1] = 1.0
    diag[:, 2, 2] = d
    return np.matmul(np.matmul(Vt.transpose(0, 2, 1), diag), U.transpose(0, 2, 1))


R = kabsch_rot_batch(pos_c_centered, ref_c_centered)   # (n_frames, 3, 3)

trace = np.trace(R, axis1=1, axis2=2)
theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))     # (n_frames,) radians

axis_raw = np.stack([R[:, 2, 1] - R[:, 1, 2],
                      R[:, 0, 2] - R[:, 2, 0],
                      R[:, 1, 0] - R[:, 0, 1]], axis=1)   # (n_frames, 3)
denom = 2 * np.sin(theta)
denom_safe = np.where(np.abs(denom) < 1e-8, 1.0, denom)   # avoid divide-by-zero for near-zero rotations
axis = axis_raw / denom_safe[:, None]
norms = np.linalg.norm(axis, axis=1, keepdims=True)
axis = axis / np.where(norms < 1e-8, 1.0, norms)

# Sign-consistency: flip each frame's axis if it points opposite to the previous frame's
for i in range(1, n_frames):
    if np.dot(axis[i], axis[i - 1]) < 0:
        axis[i] = -axis[i]
        theta[i] = -theta[i]  # keep (axis, theta) pair physically equivalent after the flip

# Mean axis via simple vector averaging + renormalization (fine for a
# reasonably concentrated set of axis directions; not a rigorous spherical
# mean, but good enough for a wander-angle sanity check)
mean_axis = axis.mean(axis=0)
mean_axis = mean_axis / np.linalg.norm(mean_axis)
wander_deg = np.degrees(np.arccos(np.clip(np.abs(axis @ mean_axis), -1, 1)))  # abs() -- axis sign is arbitrary per-frame

theta_deg = np.degrees(np.abs(theta))

out_npz = f"{OUT_DIR}/A_{condition}_rep{rep}_axis.npz"
np.savez_compressed(out_npz, time_ns=time_ns, theta_deg=theta_deg, axis=axis,
                     mean_axis=mean_axis, wander_deg=wander_deg)
print(f"[{condition} rep{rep}] n_frames={n_frames}  "
      f"theta: mean={theta_deg.mean():.2f} std={theta_deg.std():.2f}  "
      f"wander: mean={wander_deg.mean():.2f} std={wander_deg.std():.2f} max={wander_deg.max():.2f}", flush=True)
print(f"Wrote {out_npz}", flush=True)
