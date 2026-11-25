"""
Step 1: Compute time-fraction contact map from COG trajectories.
Each replicate is processed separately (parallelized via SLURM array).
Output: data_cog/contact_counts_rep{rep}.npy  (N x N integer counts)
        data_cog/contact_nframes_rep{rep}.npy  (scalar: total frames counted)
"""
import os, sys
import numpy as np
from scipy.spatial.distance import cdist

rep    = int(sys.argv[1])
ALLOD  = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(ALLOD, 'data_cog')
CUTOFF = 8.0   # Å
STRIDE = 5     # use every 5th frame for speed

npy = os.path.join(DATA, f'me1_inh_rep{rep}_cog.npy')
if not os.path.isfile(npy):
    print(f'rep{rep}: file not found — skipping'); sys.exit(0)

coords = np.load(npy)           # (frames, N, 3)
N      = coords.shape[1]
coords = coords[::STRIDE]       # stride
n_frames = len(coords)

print(f'rep{rep}: {n_frames} frames (stride={STRIDE}), N={N}', flush=True)

counts = np.zeros((N, N), dtype=np.int32)
for i, frame in enumerate(coords):
    d = cdist(frame, frame)
    counts += (d <= CUTOFF).astype(np.int32)
    if (i+1) % 200 == 0:
        print(f'  rep{rep}: frame {i+1}/{n_frames}', flush=True)

out_c = os.path.join(DATA, f'contact_counts_rep{rep}.npy')
out_n = os.path.join(DATA, f'contact_nframes_rep{rep}.npy')
np.save(out_c, counts)
np.save(out_n, np.array([n_frames]))
print(f'rep{rep}: saved {out_c}', flush=True)
