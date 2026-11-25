#!/usr/bin/env python
"""
Three-way comparison: ME1+Inhibitor, ME3+Inhibitor, ME1-noInhibitor.

Per-trajectory means are treated as independent replicates.
All pairwise Welch's t-tests + Cohen's d for each contact distance.
Also runs Levene's test (autocorrelation-corrected) for variance comparison.

FRAMES_PER_TRAJ differs per system:
  ME1+Inhibitor   : 25000  (50 ns at 2 ps/frame, 5 replicates)
  ME3+Inhibitor   : 25000  (50 ns at 2 ps/frame, 5 replicates)
  ME1-noInhibitor : 30000  (62 ns at 2 ps/frame; equil already removed in CSV)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools

from config import BASE  # set ALLOSTERY_DATA to override

CONTACTS = [
    "Arg-Malate(O1/O2)",
    "Lys-Malate(O3)",
    "Tyr-Malate(O4/O5)",
    "Asn(C1)-Malate",
    "Asn(C4)-Malate",
    "Mg-Malate(O3)",
]

SYSTEMS = {
    "ME1+Inhib":   ("malate_distances_ME1.csv",           25000, 5),
    "ME3+Inhib":   ("malate_distances_ME3.csv",           25000, 5),
    "ME1-noInhib": ("malate_distances_ME1_noInhibitor.csv", 30000, 6),
}

COLORS = {
    "ME1+Inhib":   "#2166ac",
    "ME3+Inhib":   "#d6604d",
    "ME1-noInhib": "#4dac26",
}

short = ["Arg", "Lys", "Tyr", "Asn(C1)", "Asn(C4)", "Mg"]

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def autocorr_time(x):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for k in range(1, n):
        if acf[k] < 0:
            break
        tau += acf[k]
    return max(tau, 1.0)

# ── Load per-trajectory means and within-traj subsampled series ───────────────
traj_means = {}
pooled_ind  = {}   # for Levene's test

for sys, (fname, fpt, max_traj) in SYSTEMS.items():
    df = pd.read_csv(f"{BASE}/{fname}")
    df["traj"] = (df["Frame"] // fpt).clip(upper=max_traj - 1)
    traj_means[sys] = df.groupby("traj")[CONTACTS].mean()
    n_traj = df["traj"].nunique()
    print(f"\n{sys} ({n_traj} replicates):")
    print(traj_means[sys].round(2).to_string())

    # autocorrelation subsampling per trajectory for Levene
    pooled_ind[sys] = {c: [] for c in CONTACTS}
    for tid, tdf in df.groupby("traj"):
        for col in CONTACTS:
            series = tdf[col].dropna().values
            if len(series) < 10:
                continue
            tau  = autocorr_time(series)
            step = max(1, int(np.ceil(2 * tau)))
            pooled_ind[sys][col].extend(series[::step])

# ── Pairwise Welch's t-test on per-trajectory means ──────────────────────────
pairs = list(itertools.combinations(SYSTEMS.keys(), 2))

print(f"\n{'='*90}")
print("MEAN CONTACT DISTANCES — pairwise Welch's t-test (n = replicates per system)")
print(f"{'='*90}")

all_results = {}
for col in CONTACTS:
    print(f"\n  {col}")
    print(f"  {'Pair':<30} {'A mean':>8} {'B mean':>8} {'diff':>7} {'Cohen d':>8} {'p':>10} sig")
    print(f"  {'-'*75}")
    all_results[col] = []
    for sA, sB in pairs:
        a = traj_means[sA][col].values
        b = traj_means[sB][col].values
        diff = np.mean(b) - np.mean(a)
        d    = cohens_d(a, b)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        all_results[col].append(dict(pair=f"{sA} vs {sB}", sA=sA, sB=sB,
                                     mA=np.mean(a), mB=np.mean(b),
                                     diff=diff, d=d, p=p, sig=sig))
        print(f"  {sA+' vs '+sB:<30} {np.mean(a):>8.2f} {np.mean(b):>8.2f} "
              f"{diff:>+7.2f} {d:>8.2f} {p:>10.4f} {sig}")

# ── Levene's test on autocorrelation-corrected samples ───────────────────────
print(f"\n{'='*90}")
print("FLUCTUATION (Levene's test, autocorrelation-corrected)")
print(f"{'='*90}")
lev_results = {}
sys_list = list(SYSTEMS.keys())
for col in CONTACTS:
    print(f"\n  {col}")
    print(f"  {'Pair':<30} {'A std':>8} {'B std':>8} {'ratio':>7} {'Lev F':>10} {'p':>10} sig")
    print(f"  {'-'*75}")
    lev_results[col] = []
    for sA, sB in pairs:
        a = np.array(pooled_ind[sA][col])
        b = np.array(pooled_ind[sB][col])
        sA_std = np.std(a, ddof=1)
        sB_std = np.std(b, ddof=1)
        ratio  = sB_std / sA_std if sA_std > 0 else np.nan
        F, p   = stats.levene(a, b, center='median')
        sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lev_results[col].append(dict(pair=f"{sA} vs {sB}", sA=sA, sB=sB,
                                     stdA=sA_std, stdB=sB_std,
                                     ratio=ratio, F=F, p=p, sig=sig))
        print(f"  {sA+' vs '+sB:<30} {sA_std:>8.3f} {sB_std:>8.3f} "
              f"{ratio:>7.2f} {F:>10.3f} {p:>10.4f} {sig}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: mean distances per system (mean ± std across replicates)
ax = axes[0, 0]
x  = np.arange(len(CONTACTS))
w  = 0.25
offsets = np.array([-1, 0, 1]) * w
for i, (sys, _) in enumerate(SYSTEMS.items()):
    means = traj_means[sys][CONTACTS].mean().values
    errs  = traj_means[sys][CONTACTS].std().values
    ax.bar(x + offsets[i], means, w, yerr=errs, capsize=3,
           label=sys, color=COLORS[sys], alpha=0.85, edgecolor="k", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel("Mean Distance (Å)", fontsize=11)
ax.set_title("A — Mean contact distances\n(mean ± std across replicates)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Top-right: std dev per system
ax = axes[0, 1]
for i, (sys, _) in enumerate(SYSTEMS.items()):
    stds = [np.std(pooled_ind[sys][c], ddof=1) for c in CONTACTS]
    ax.bar(x + offsets[i], stds, w,
           label=sys, color=COLORS[sys], alpha=0.85, edgecolor="k", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel("Std Dev of Distance (Å)", fontsize=11)
ax.set_title("B — Contact fluctuation\n(autocorr-corrected independent samples)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Bottom-left: Cohen's d heatmap for mean distances
pair_labels = [f"{r['sA'].split('+')[0].split('-')[0]}\nvs\n{r['sB'].split('+')[0].split('-')[0]}"
               for r in all_results[CONTACTS[0]]]
pair_labels_short = [f"{r['sA']} vs {r['sB']}" for r in all_results[CONTACTS[0]]]
d_matrix = np.array([[r['d'] for r in all_results[col]] for col in CONTACTS])
ax = axes[1, 0]
im = ax.imshow(np.abs(d_matrix), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([f"{sA.replace('+Inhib','').replace('-noInhib','-noI')}\nvs\n"
                    f"{sB.replace('+Inhib','').replace('-noInhib','-noI')}"
                    for sA, sB in pairs], fontsize=8)
ax.set_yticks(range(len(CONTACTS))); ax.set_yticklabels(short, fontsize=9)
ax.set_title("C — |Cohen's d| for mean distances\n(>0.8 = large effect)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="|Cohen's d|")
for i, col in enumerate(CONTACTS):
    for j, r in enumerate(all_results[col]):
        ax.text(j, i, r['sig'], ha='center', va='center', fontsize=9,
                color='white' if abs(r['d']) > 1.5 else 'black')

# Bottom-right: Levene ratio heatmap
lev_matrix  = np.array([[r['ratio'] for r in lev_results[col]] for col in CONTACTS])
sig_matrix  = [[r['sig']   for r in lev_results[col]] for col in CONTACTS]
ax = axes[1, 1]
im2 = ax.imshow(lev_matrix, cmap="RdYlBu_r", aspect="auto", vmin=0.5, vmax=2.5)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([f"{sA.replace('+Inhib','').replace('-noInhib','-noI')}\nvs\n"
                    f"{sB.replace('+Inhib','').replace('-noInhib','-noI')}"
                    for sA, sB in pairs], fontsize=8)
ax.set_yticks(range(len(CONTACTS))); ax.set_yticklabels(short, fontsize=9)
ax.set_title("D — Levene fluctuation ratio (B std / A std)\n(red = B fluctuates more)", fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=ax, label="Std ratio (B/A)")
for i, col in enumerate(CONTACTS):
    for j, r in enumerate(lev_results[col]):
        ax.text(j, i, r['sig'], ha='center', va='center', fontsize=9,
                color='white' if lev_matrix[i, j] > 2.0 else 'black')

plt.suptitle("Three-Way Comparison: ME1+Inhibitor vs ME3+Inhibitor vs ME1-noInhibitor\n"
             "Malate active site contact distances",
             fontsize=13, y=1.01)
plt.tight_layout()

out = f"{BASE}/fig8_threeway_stats.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved {out}")
