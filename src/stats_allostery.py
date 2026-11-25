#!/usr/bin/env python
"""
Proper per-trajectory statistics on malate contact distances.
Each of the 5 trajectories is treated as one independent replicate.
Compares ME1 vs ME3 with Welch's t-test and Cohen's d effect size.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from config import BASE  # set ALLOSTERY_DATA to override

CONTACTS = [
    "Arg-Malate(O1/O2)",
    "Lys-Malate(O3)",
    "Tyr-Malate(O4/O5)",
    "Asn(C1)-Malate",
    "Asn(C4)-Malate",
    "Mg-Malate(O3)",
]

FRAMES_PER_TRAJ = 25000  # 50 ns at 2 ps/frame

def assign_trajectory(frame_col):
    """Map global frame index back to trajectory number 0-4."""
    return (frame_col // FRAMES_PER_TRAJ).clip(upper=4)

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

# ── Load and assign trajectories ─────────────────────────────────────────────
traj_means = {}
for sys in ["ME1", "ME3"]:
    df = pd.read_csv(f"{BASE}/malate_distances_{sys}.csv")
    df["traj"] = assign_trajectory(df["Frame"])
    # Per-trajectory mean for each contact
    traj_means[sys] = df.groupby("traj")[CONTACTS].mean()
    print(f"\n{sys} — per-trajectory means:")
    print(traj_means[sys].round(2).to_string())
    print(f"  frames per traj: {df.groupby('traj').size().to_dict()}")

# ── Stats: Welch's t-test + Cohen's d, ME1 vs ME3 ────────────────────────────
print(f"\n{'='*75}")
print(f"{'Contact':<25} {'ME1 mean':>9} {'ME3 mean':>9} {'diff':>7} "
      f"{'Cohen d':>8} {'p-value':>10} {'sig':>5}")
print(f"{'='*75}")

results = []
for col in CONTACTS:
    me1_vals = traj_means["ME1"][col].values
    me3_vals = traj_means["ME3"][col].values

    diff = np.mean(me3_vals) - np.mean(me1_vals)
    d = cohens_d(me1_vals, me3_vals)
    t, p = stats.ttest_ind(me1_vals, me3_vals, equal_var=False)  # Welch's

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    results.append(dict(contact=col, me1=np.mean(me1_vals), me3=np.mean(me3_vals),
                        diff=diff, d=d, p=p, sig=sig))
    print(f"{col:<25} {np.mean(me1_vals):>9.2f} {np.mean(me3_vals):>9.2f} "
          f"{diff:>+7.2f} {d:>8.2f} {p:>10.4f} {sig:>5}")

print(f"\nCohen's d interpretation: small=0.2, medium=0.5, large=0.8")

# ── Same for std dev (fluctuation) ───────────────────────────────────────────
traj_stds = {}
for sys in ["ME1", "ME3"]:
    df = pd.read_csv(f"{BASE}/malate_distances_{sys}.csv")
    df["traj"] = assign_trajectory(df["Frame"])
    traj_stds[sys] = df.groupby("traj")[CONTACTS].std()

print(f"\n{'='*75}")
print("FLUCTUATION (std dev) — per-trajectory, then compared ME1 vs ME3")
print(f"{'='*75}")
print(f"{'Contact':<25} {'ME1 std':>9} {'ME3 std':>9} {'ratio':>7} "
      f"{'Cohen d':>8} {'p-value':>10} {'sig':>5}")
print(f"{'='*75}")

std_results = []
for col in CONTACTS:
    me1_stds = traj_stds["ME1"][col].values
    me3_stds = traj_stds["ME3"][col].values

    ratio = np.mean(me3_stds) / np.mean(me1_stds) if np.mean(me1_stds) > 0 else np.nan
    d = cohens_d(me1_stds, me3_stds)
    t, p = stats.ttest_ind(me1_stds, me3_stds, equal_var=False)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    std_results.append(dict(contact=col, me1=np.mean(me1_stds), me3=np.mean(me3_stds),
                            ratio=ratio, d=d, p=p, sig=sig))
    print(f"{col:<25} {np.mean(me1_stds):>9.2f} {np.mean(me3_stds):>9.2f} "
          f"{ratio:>7.2f} {d:>8.2f} {p:>10.4f} {sig:>5}")

# ── Plot: mean ± std across 5 replicates ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"ME1": "#2166ac", "ME3": "#d6604d"}
x = np.arange(len(CONTACTS))
w = 0.35

for ax, (metric, title, ylabel, data_dict) in zip(axes, [
    ("mean",  "Mean Contact Distance\n(mean ± std across 5 replicates)",
     "Distance (Å)", traj_means),
    ("std",   "Contact Fluctuation (Std Dev)\n(mean ± std across 5 replicates)",
     "Std Dev of Distance (Å)", traj_stds),
]):
    for i, (sys, df) in enumerate(data_dict.items()):
        means = df[CONTACTS].mean().values
        errs  = df[CONTACTS].std().values
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, means, w, yerr=errs, capsize=4,
                      label=sys, color=colors[sys], alpha=0.85,
                      edgecolor="k", linewidth=0.7)

    # Significance markers
    res = results if metric == "mean" else std_results
    for j, r in enumerate(res):
        if r["sig"] != "ns":
            ymax = max(data_dict["ME1"][CONTACTS[j]].mean(),
                       data_dict["ME3"][CONTACTS[j]].mean())
            yerr_max = max(data_dict["ME1"][CONTACTS[j]].std(),
                           data_dict["ME3"][CONTACTS[j]].std())
            ax.text(x[j], ymax + yerr_max + 0.15, r["sig"],
                    ha="center", fontsize=11, fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(CONTACTS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)

plt.suptitle("ME1 vs ME3 Malate Contact Statistics (n=5 replicates each)",
             fontsize=13, y=1.02)
plt.tight_layout()
out = f"{BASE}/fig4_stats_per_replicate.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved {out}")
