#!/usr/bin/env python
"""
Summarize + plot the open/closed-state and inhibitor-stability metrics
computed by compute_openclosed_metrics.py, comparing ME1+LIGAND (with_inhibitor)
vs ME1 apo (no_inhibitor) across all 5 replicates each.

Reads: openclosed_metrics/*_metrics.csv
Writes: openclosed_summary.txt, openclosed_summary.pdf
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import BASE  # set ALLOSTERY_DATA to override
METRICS_DIR = f"{BASE}/openclosed_metrics"

SYSTEMS = {
    'with_inhibitor': 'me1_239_malate_restrained',
    'no_inhibitor':   'me1_malate_noinhibit_restrained',
}

def load_system(prefix):
    dfs = []
    for rep in range(5):
        f = f"{METRICS_DIR}/{prefix}_{rep}_metrics.csv"
        df = pd.read_csv(f)
        df['rep'] = rep
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

data = {sysname: load_system(prefix) for sysname, prefix in SYSTEMS.items()}

lines = []
def log(s=''):
    print(s)
    lines.append(s)

log("=" * 78)
log("OPEN/CLOSED-STATE METRICS SUMMARY -- ME1 +LIGAND vs ME1 apo")
log("=" * 78)
log(f"Frames per system: with_inhibitor={len(data['with_inhibitor'])}  "
    f"no_inhibitor={len(data['no_inhibitor'])}  (5 reps x 100,000 frames each)")

# ── 1. Malate restraint sanity check ──────────────────────────────────────────
log("\n--- 1. Malate flat-bottom restraint check (rmax=15A) ---")
for sysname, df in data.items():
    d = df['malate_restraint']
    n_viol = (d > 15.0).sum()
    log(f"  {sysname:15s}: mean={d.mean():.2f}  std={d.std():.2f}  "
        f"max={d.max():.2f}  frames>15A: {n_viol}/{len(d)} ({100*n_viol/len(d):.3f}%)")

# ── 2. Inhibitor pocket residence (with_inhibitor only) ───────────────────────
log("\n--- 2. Inhibitor (LIGAND) pocket residence (with_inhibitor only) ---")
df = data['with_inhibitor']
for rep in range(5):
    d = df[df.rep == rep]['inhib_pocket_dist']
    log(f"  rep {rep}: mean={d.mean():.2f}  std={d.std():.2f}  max={d.max():.2f} A")
d = df['inhib_pocket_dist']
log(f"  ALL REPS: mean={d.mean():.2f}  std={d.std():.2f}  max={d.max():.2f} A  "
    f"(frames>10A from pocket centroid: {100*(d>10).sum()/len(d):.2f}%)")

# ── 3. Open/closed catalytic distance metrics ─────────────────────────────────
CLOSED_METRICS = {
    'lys183_malate':      ('Lys183(NZ)-malate(OH)', 2.5, 3.0, '+2.0 or more (open/parked)'),
    'asp278_glu255':      ('Asp278-Glu255 "lock"',  None, None, 'forms in closed state'),
    'tyr112_malate_c3':   ('Tyr112(OH)-malate(C3)', 3.1, 3.1, 'poised over C3 in closed'),
    'arg165_malate':      ('Arg165-malate(C1-carboxylate)', None, None, 'tight in closed'),
    'asn418_malate':      ('Asn418-malate(C1-carboxylate)', None, None, 'tight in closed'),
}

log("\n--- 3. Open/closed catalytic distances (mean +/- std over all 5 reps, all frames) ---")
log(f"{'metric':32s} {'with_inhibitor':>20s} {'no_inhibitor':>20s}  {'lit. closed-state value':>28s}")
for col, (label, lo, hi, note) in CLOSED_METRICS.items():
    row = f"{label:32s}"
    for sysname in SYSTEMS:
        d = data[sysname][col]
        row += f" {d.mean():7.2f}+/-{d.std():5.2f} A "
    row += f"  {note}"
    log(row)

# ── 4. Mg2+ coordination ──────────────────────────────────────────────────────
log("\n--- 4. Mg2+ coordination distances (mean +/- std) ---")
for col, label in [('mg_glu255', 'Mg-Glu255'), ('mg_asp256', 'Mg-Asp256'),
                    ('mg_asp279', 'Mg-Asp279'), ('mg_malate', 'Mg-malate(nearest O)')]:
    row = f"{label:24s}"
    for sysname in SYSTEMS:
        d = data[sysname][col]
        row += f" {d.mean():6.2f}+/-{d.std():5.2f} A "
    log(row)

# ── 5. LIGAND ring-stacking distances (with_inhibitor only) ──────────────────────
log("\n--- 5. LIGAND aromatic-ring-centroid distances to Phe263/His321 (with_inhibitor only) ---")
log("    (LIGAND has two separate ring systems, ring1=simple phenyl, ring2=fused")
log("     heterocycle -- see compute_openclosed_metrics.py docstring; not a claim")
log("     about which specific ring stacks with which residue)")
df = data['with_inhibitor']
for col, label in [('ligand_ring1_phe263', 'LIGAND ring1 - Phe263'),
                    ('ligand_ring1_his321', 'LIGAND ring1 - His321'),
                    ('ligand_ring2_phe263', 'LIGAND ring2 - Phe263'),
                    ('ligand_ring2_his321', 'LIGAND ring2 - His321')]:
    d = df[col]
    log(f"  {label:22s}: mean={d.mean():.2f}  std={d.std():.2f}  min={d.min():.2f}  max={d.max():.2f} A")

with open(f"{BASE}/openclosed_summary.txt", 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\nWrote text summary -> {BASE}/openclosed_summary.txt")

# ══════════════════════════════════════════════════════════════════════════════
# Figure: distributions for the 5 key open/closed metrics + inhibitor pocket dist
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_cols = ['lys183_malate', 'asp278_glu255', 'tyr112_malate_c3',
             'arg165_malate', 'asn418_malate', 'inhib_pocket_dist']
titles = ['Lys183(NZ)-malate(OH)\n[closed: 2.5-3.0 A]',
          'Asp278-Glu255 "lock"',
          'Tyr112(OH)-malate(C3)\n[closed: ~3.1 A]',
          'Arg165-malate(C1-carboxylate)',
          'Asn418-malate(C1-carboxylate)',
          'LIGAND - pocket centroid\n(with_inhibitor only)']
colors = {'with_inhibitor': '#d62728', 'no_inhibitor': '#1f77b4'}

for ax, col, title in zip(axes.flat, plot_cols, titles):
    for sysname, df in data.items():
        if col == 'inhib_pocket_dist' and sysname != 'with_inhibitor':
            continue
        d = df[col].values
        ax.hist(d, bins=100, density=True, alpha=0.5, color=colors[sysname],
                label=sysname)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('distance (A)')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)

fig.suptitle('ME1 +LIGAND (open-locked hypothesis) vs ME1 apo -- '
              'open/closed catalytic-distance distributions (5 reps x 1us each)',
              fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{BASE}/openclosed_summary.pdf")
print(f"Wrote figure -> {BASE}/openclosed_summary.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Figure: time series (mean across reps, per ns bin) for lys183_malate --
# the primary literature metric -- to see drift/bimodality over the full 1us
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 5))
for sysname, df in data.items():
    binned = df.groupby(df['time_ns'].round(0))['lys183_malate'].mean()
    ax2.plot(binned.index, binned.values, label=sysname, color=colors[sysname], lw=1)
ax2.axhline(3.0, color='gray', ls='--', lw=1, label='closed-state cutoff (~3.0 A)')
ax2.set_xlabel('time (ns)')
ax2.set_ylabel('Lys183(NZ)-malate(OH) distance (A), mean over 5 reps')
ax2.set_title('Lys183-malate distance over the full 1us trajectory')
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{BASE}/openclosed_lys183_timeseries.pdf")
print(f"Wrote figure -> {BASE}/openclosed_lys183_timeseries.pdf")
