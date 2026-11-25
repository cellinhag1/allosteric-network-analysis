"""
Per-residue RMSF t-test: holo vs apo across replicates.
Computes RMSF for every replicate individually, then Welch's t-test per residue.
Saves: rmsf_ttest_pvals.npy, rmsf_ttest_reject.npy, rmsf_holo_all.npy, rmsf_apo_all.npy
"""
import os, sys
import numpy as np
from scipy import stats

ALLOD   = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(ALLOD, 'data_cog')
WINDOW  = 'last25ns'        # frame cutoff: keep frames where frame_idx >= 5000
N_REPS  = 25

def compute_rmsf_rep(cond, rep, start_frame=5000):
    npy = os.path.join(DATA, f'{cond}_rep{rep}_cog.npy')
    if not os.path.isfile(npy):
        return None
    coords = np.load(npy).astype(np.float32)
    idx_f  = os.path.join(DATA, f'{cond}_rep{rep}_frame_idx.npy')
    if os.path.isfile(idx_f):
        fidx   = np.load(idx_f)
        coords = coords[fidx >= start_frame]
    else:
        coords = coords[start_frame:]
    if len(coords) < 50:
        print(f'  {cond} rep{rep}: only {len(coords)} frames — skipping')
        return None
    disp = coords - coords.mean(0)
    return np.sqrt((disp**2).sum(-1).mean(0)).astype(np.float32)   # (N,)

print('Computing per-replicate RMSF (COG, last25ns)...', flush=True)

for cond_inh, cond_apo in [('me1_inh', 'me1_noinh')]:
    rmsf_inh, rmsf_apo = [], []
    for rep in range(N_REPS):
        r = compute_rmsf_rep(cond_inh, rep)
        if r is not None:
            rmsf_inh.append(r)
            print(f'  {cond_inh} rep{rep}: mean RMSF={r.mean():.3f} Å', flush=True)
    for rep in range(N_REPS):
        r = compute_rmsf_rep(cond_apo, rep)
        if r is not None:
            rmsf_apo.append(r)
            print(f'  {cond_apo} rep{rep}: mean RMSF={r.mean():.3f} Å', flush=True)

    arr_inh = np.stack(rmsf_inh)   # (n_inh, N)
    arr_apo = np.stack(rmsf_apo)   # (n_apo, N)
    print(f'\n  n_inh={len(rmsf_inh)}, n_apo={len(rmsf_apo)}, N_residues={arr_inh.shape[1]}', flush=True)

    _, pvals = stats.ttest_ind(arr_inh, arr_apo, axis=0, equal_var=False)

    # Benjamini-Hochberg over N_residues tests (much less multiple testing than the MI case)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    thresholds = 0.05 * np.arange(1, n+1) / n
    below = ranked_p <= thresholds
    reject = np.zeros(n, dtype=bool)
    if below.any():
        mk = np.where(below)[0].max()
        rej_ordered = np.zeros(n, dtype=bool)
        rej_ordered[:mk+1] = True
        reject[order] = rej_ordered

    n_sig = reject.sum()
    print(f'\n  FDR q<0.05: {n_sig}/{n} residues significantly different RMSF', flush=True)

    resids = np.load(os.path.join(DATA, f'{cond_inh}_resids.npy'))
    if n_sig > 0:
        sig_resids = resids[reject]
        delta_rmsf = arr_inh.mean(0) - arr_apo.mean(0)
        sig_delta  = delta_rmsf[reject]
        order_d    = np.argsort(np.abs(sig_delta))[::-1]
        print(f'\n  Top significant residues by |ΔRMSF|:', flush=True)
        print(f'  {"Resid":>6}  {"RMSF holo":>10}  {"RMSF apo":>10}  {"ΔRMSF":>9}  p-val')
        for k in order_d[:20]:
            ri = np.where(resids == sig_resids[order_d[k]])[0][0]
            print(f'  {int(sig_resids[order_d[k]]):>6d}  '
                  f'{arr_inh.mean(0)[ri]:>10.4f}  '
                  f'{arr_apo.mean(0)[ri]:>10.4f}  '
                  f'{delta_rmsf[ri]:>+9.4f}  '
                  f'{pvals[ri]:.2e}', flush=True)

    # Save outputs
    np.save(os.path.join(DATA, f'{cond_inh}_vs_{cond_apo}_rmsf_holo_all.npy'), arr_inh)
    np.save(os.path.join(DATA, f'{cond_inh}_vs_{cond_apo}_rmsf_apo_all.npy'),  arr_apo)
    np.save(os.path.join(DATA, f'{cond_inh}_vs_{cond_apo}_rmsf_pvals.npy'),    pvals)
    np.save(os.path.join(DATA, f'{cond_inh}_vs_{cond_apo}_rmsf_reject.npy'),   reject)
    print(f'\n  Saved rmsf_holo_all, rmsf_apo_all, rmsf_pvals, rmsf_reject to {DATA}', flush=True)

print('Done.', flush=True)
