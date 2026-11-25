#!/usr/bin/env python
"""
Plot inhibitor (LIGAND) stability: ME1 vs ME3 over 1 µs.

Each panel shows mean ± SD across all 5 replicates, with individual
replicate traces as faint background lines for context.

Output: inhibitor_stability_1us.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import BASE  # set ALLOSTERY_DATA to override
OUT      = f"{BASE}/dynamics/fixed_dcd"
SYSTEMS  = ['me1', 'me3']
PREFIXES = {'me1': 'me1_239_malate', 'me3': 'me3_239_malate'}
LABELS   = {'me1': 'ME1 + LIGAND', 'me3': 'ME3 + LIGAND'}
COLORS   = {'me1': '#2166ac', 'me3': '#d6604d'}   # blue / red
N_REPS   = 5
DT_NS    = 0.01    # 10 ps / frame → 0.01 ns
SMOOTH   = 100     # running-average window = 1 ns


def running_mean(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def load_reps(prefix, metric):
    """Load .npy arrays for all reps; return (arrays_list, max_len)."""
    arrays = []
    for rep in range(N_REPS):
        f = f"{OUT}/{prefix}_rep{rep}_{metric}.npy"
        if os.path.isfile(f):
            arrays.append(np.load(f))
        else:
            print(f"  Missing: {f}")
    return arrays


def mean_std(arrays):
    """Pad shorter arrays with NaN, return (mean, std, time_ns)."""
    if not arrays:
        return None, None, None
    max_len = max(len(a) for a in arrays)
    mat = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        mat[i, :len(a)] = a
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0), np.arange(max_len) * DT_NS


fig, (ax_com, ax_sasa) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
fig.suptitle('LIGAND Inhibitor Stability — ME1 vs ME3 (1 µs MD, n=5 replicates)',
             fontsize=13, fontweight='bold')

for sys in SYSTEMS:
    prefix = PREFIXES[sys]
    color  = COLORS[sys]
    label  = LABELS[sys]

    # ── COM distance ──────────────────────────────────────────────────────────
    com_arrs = load_reps(prefix, 'com_dist')
    mean_c, std_c, t_c = mean_std(com_arrs)

    if mean_c is not None:
        # Individual reps — faint background
        for arr in com_arrs:
            t_r = np.arange(len(arr)) * DT_NS
            ax_com.plot(t_r, arr, color=color, alpha=0.12, lw=0.6)

        # Smoothed mean ± SD — primary
        sm_mean = running_mean(mean_c, SMOOTH)
        sm_std  = running_mean(std_c,  SMOOTH)
        t_sm    = np.arange(len(sm_mean)) * DT_NS + (SMOOTH / 2) * DT_NS
        ax_com.plot(t_sm, sm_mean, color=color, lw=2.0, label=label)
        ax_com.fill_between(t_sm,
                            sm_mean - sm_std,
                            sm_mean + sm_std,
                            color=color, alpha=0.25)

    # ── SASA ──────────────────────────────────────────────────────────────────
    sasa_arrs = load_reps(prefix, 'sasa')
    mean_s, std_s, t_s = mean_std(sasa_arrs)

    if mean_s is not None:
        for arr in sasa_arrs:
            t_r = np.arange(len(arr)) * DT_NS
            ax_sasa.plot(t_r, arr, color=color, alpha=0.12, lw=0.6)

        sm_mean = running_mean(mean_s, SMOOTH)
        sm_std  = running_mean(std_s,  SMOOTH)
        t_sm    = np.arange(len(sm_mean)) * DT_NS + (SMOOTH / 2) * DT_NS
        ax_sasa.plot(t_sm, sm_mean, color=color, lw=2.0, label=label)
        ax_sasa.fill_between(t_sm,
                             sm_mean - sm_std,
                             sm_mean + sm_std,
                             color=color, alpha=0.25)

# ── COM panel formatting ───────────────────────────────────────────────────────
ax_com.axhline(10.0, color='black', ls='--', lw=1.0, alpha=0.5,
               label='10 Å reference')
ax_com.set_ylabel('LIGAND–Pocket COM Distance (Å)', fontsize=11)
ax_com.set_xlabel('Time (ns)', fontsize=11)
ax_com.set_ylim(bottom=0)
ax_com.legend(fontsize=10)
ax_com.set_title('Center-of-Mass Distance: LIGAND vs Allosteric Pocket', fontsize=11)

# ── SASA panel formatting ──────────────────────────────────────────────────────
ax_sasa.set_ylabel('LIGAND SASA (Ų)', fontsize=11)
ax_sasa.set_xlabel('Time (ns)', fontsize=11)
ax_sasa.set_ylim(bottom=0)
ax_sasa.legend(fontsize=10)
ax_sasa.set_title('Solvent Accessible Surface Area of LIGAND', fontsize=11)

# ── Shared note ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         'Shaded region = mean ± SD across 5 replicates  |  Thin lines = individual replicates  |  Smoothing = 1 ns running average',
         ha='center', fontsize=8, color='grey')

plt.tight_layout(rect=[0, 0.03, 1, 1])
out_fig = f"{BASE}/inhibitor_stability_1us.png"
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"Saved: {out_fig}")
