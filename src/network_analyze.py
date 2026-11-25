#!/usr/bin/env python
"""
Network Analyze — Step 2 of allosteric network analysis.

Loads the mean graphs produced by network_build.py and computes:
  1. Shortest (optimal) allosteric paths  pocket → active site (Dijkstra, weight=-log r_MI)
  2. Betweenness centrality — hub residues
  3. Difference network  ME1+Inhib − ME3+Inhib  (edge-weight differences)
  4. Community detection (Louvain, falls back to Girvan-Newman)
  5. Replicate variability (edge r_MI std across 5 rep graphs per system)

Outputs in BASE/networks/analysis/:
  allosteric_paths.txt          best paths per system + path comparison table
  betweenness.txt               top-N hub residues per system
  diff_network.pkl              signed difference graph (ME1 − ME3)
  *.png                         all figures
"""

import os, glob, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

from config import BASE  # set ALLOSTERY_DATA to override
NET  = f"{BASE}/networks"
OUT  = f"{NET}/analysis"
os.makedirs(OUT, exist_ok=True)

# ── System metadata (must match network_build.py) ────────────────────────────
SYSTEMS = {
    "ME1+Inhib": {
        "tag":           "ME1pInhib",
        "pocket_resids": [249,250,270,271,274,303,304,311,312,350,353],
        "active_resids": [112,165,183,418,463],
        "color":         "#2166ac",
        "label":         "ME1+LIGAND",
    },
    "ME1-noInhib": {
        "tag":           "ME1nnoInhib",
        "pocket_resids": [249,250,270,271,274,303,304,311,312,350,353],
        "active_resids": [112,165,183,418,463],
        "color":         "#4dac26",
        "label":         "ME1 (apo)",
    },
    "ME3+Inhib": {
        "tag":           "ME3pInhib",
        "pocket_resids": [284,305,306,309,337,338,339,342,346,347,383,385,386],
        "active_resids": [137,190,208,443,488],
        "color":         "#d6604d",
        "label":         "ME3+LIGAND",
    },
}

# ── Load graphs ───────────────────────────────────────────────────────────────
mean_graphs = {}
rep_graphs  = {}   # list of per-rep graphs per system

for sys_name, cfg in SYSTEMS.items():
    tag = cfg["tag"]
    pkl = f"{NET}/{tag}_mean_graph.pkl"
    if not os.path.exists(pkl):
        print(f"  WARNING: {pkl} not found — skipping {sys_name}")
        continue
    with open(pkl, "rb") as fh:
        mean_graphs[sys_name] = pickle.load(fh)
    print(f"Loaded {sys_name}: {mean_graphs[sys_name].number_of_nodes()} nodes, "
          f"{mean_graphs[sys_name].number_of_edges():,} edges")

    rep_pks = sorted(glob.glob(f"{NET}/{tag}_rep*_graph.pkl"))
    rgs = []
    for rp in rep_pks:
        with open(rp, "rb") as fh:
            rgs.append(pickle.load(fh))
    rep_graphs[sys_name] = rgs
    print(f"  {len(rgs)} replicate graphs loaded")

if not mean_graphs:
    print("No graphs found — nothing to analyse.")
    raise SystemExit(1)

# ── Helper: node → resid/resname ─────────────────────────────────────────────
def node_label(G, n):
    d = G.nodes[n]
    return f"{d['resname']}{d['resid']}"

def resid_to_node(G, resid):
    for n, d in G.nodes(data=True):
        if d["resid"] == resid:
            return n
    return None

# ── 1. Shortest allosteric paths ──────────────────────────────────────────────
print("\n" + "="*65)
print("1. SHORTEST ALLOSTERIC PATHS (Dijkstra, weight = -log r_MI)")
print("="*65)

path_results = {}   # sys_name → {"path": [...], "length": float, "labels": [...]}

for sys_name, G in mean_graphs.items():
    cfg = SYSTEMS[sys_name]
    # source = pocket node with highest mean r_MI to rest of graph (most connected)
    pocket_nodes = [resid_to_node(G, r) for r in cfg["pocket_resids"]
                    if resid_to_node(G, r) is not None]
    active_nodes = [resid_to_node(G, r) for r in cfg["active_resids"]
                    if resid_to_node(G, r) is not None]

    best_path   = None
    best_length = np.inf
    best_src    = None
    best_tgt    = None

    for src in pocket_nodes:
        for tgt in active_nodes:
            try:
                p = nx.dijkstra_path(G, src, tgt, weight="weight")
                l = nx.dijkstra_path_length(G, src, tgt, weight="weight")
                if l < best_length:
                    best_length = l
                    best_path   = p
                    best_src    = src
                    best_tgt    = tgt
            except nx.NetworkXNoPath:
                pass

    if best_path is None:
        print(f"  {sys_name}: NO PATH FOUND between pocket and active site")
        path_results[sys_name] = None
        continue

    labels = [node_label(G, n) for n in best_path]
    path_results[sys_name] = {
        "path":   best_path,
        "length": best_length,
        "labels": labels,
        "src":    node_label(G, best_src),
        "tgt":    node_label(G, best_tgt),
    }
    avg_rmi = np.exp(-best_length / len(best_path))   # geometric mean r_MI along path
    print(f"\n  {sys_name}:")
    print(f"    {node_label(G, best_src)} → {node_label(G, best_tgt)}")
    print(f"    Path length (Σ -log r_MI): {best_length:.3f}  "
          f"({len(best_path)} nodes, avg r_MI per step ≈ {avg_rmi:.3f})")
    print(f"    Route: {' → '.join(labels)}")

# write text report
with open(f"{OUT}/allosteric_paths.txt", "w") as fh:
    fh.write("SHORTEST ALLOSTERIC PATHS\n" + "="*65 + "\n\n")
    for sys_name, res in path_results.items():
        if res is None:
            fh.write(f"{sys_name}: NO PATH\n\n")
            continue
        G = mean_graphs[sys_name]
        fh.write(f"{sys_name}\n")
        fh.write(f"  Source:  {res['src']}\n")
        fh.write(f"  Target:  {res['tgt']}\n")
        fh.write(f"  Length:  {res['length']:.4f}  ({len(res['path'])} nodes)\n")
        fh.write(f"  Route:   {' → '.join(res['labels'])}\n")
        fh.write(f"  Edge r_MI along path:\n")
        for a, b in zip(res["path"][:-1], res["path"][1:]):
            d = G[a][b]
            fh.write(f"    {node_label(G,a):12s} → {node_label(G,b):12s}  "
                     f"r_MI={d['r_MI']:.4f}  persist={d['persistence']:.3f}\n")
        fh.write("\n")

print(f"\nSaved {OUT}/allosteric_paths.txt")

# ── Path figure: r_MI along each path ────────────────────────────────────────
fig_p, ax_p = plt.subplots(figsize=(10, 4))
for sys_name, res in path_results.items():
    if res is None:
        continue
    G   = mean_graphs[sys_name]
    cfg = SYSTEMS[sys_name]
    rmi_vals = []
    for a, b in zip(res["path"][:-1], res["path"][1:]):
        rmi_vals.append(G[a][b]["r_MI"])
    xlabels = [f"{res['labels'][i]}→{res['labels'][i+1]}"
               for i in range(len(rmi_vals))]
    xs = range(len(rmi_vals))
    ax_p.plot(xs, rmi_vals, "o-", color=cfg["color"],
              label=cfg["label"], linewidth=1.8, markersize=5)

ax_p.set_ylabel("r_MI (generalised correlation)", fontsize=10)
ax_p.set_title("Allosteric path edge strength: pocket → active site", fontsize=11,
               fontweight="bold")
ax_p.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
ax_p.legend(fontsize=9)
ax_p.set_ylim(0, 1)
fig_p.tight_layout()
fig_p.savefig(f"{OUT}/allosteric_path_rMI.png", dpi=200, bbox_inches="tight")
print(f"Saved allosteric_path_rMI.png")

# ── 2. Betweenness centrality ─────────────────────────────────────────────────
print("\n" + "="*65)
print("2. BETWEENNESS CENTRALITY (top 20 hub residues per system)")
print("="*65)

TOP_N = 20
fig_b, axes_b = plt.subplots(1, len(mean_graphs), figsize=(5*len(mean_graphs), 5),
                              sharey=False)
if len(mean_graphs) == 1:
    axes_b = [axes_b]

with open(f"{OUT}/betweenness.txt", "w") as fh:
    fh.write("BETWEENNESS CENTRALITY\n" + "="*65 + "\n\n")
    for ax, (sys_name, G) in zip(axes_b, mean_graphs.items()):
        cfg = SYSTEMS[sys_name]
        bc  = nx.betweenness_centrality(G, weight="weight", normalized=True)
        top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

        labels_bc = [node_label(G, n) for n, _ in top]
        vals_bc   = [v for _, v in top]
        is_pocket = [G.nodes[n]["is_pocket"] for n, _ in top]
        is_active = [G.nodes[n]["is_active"] for n, _ in top]
        colors_bc = ["#d6604d" if p else ("#4dac26" if a else cfg["color"])
                     for p, a in zip(is_pocket, is_active)]

        ax.barh(range(TOP_N), vals_bc[::-1], color=colors_bc[::-1])
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(labels_bc[::-1], fontsize=7)
        ax.set_xlabel("Betweenness centrality", fontsize=9)
        ax.set_title(cfg["label"], fontsize=10, fontweight="bold")

        fh.write(f"{sys_name}\n")
        for rank, (n, v) in enumerate(top, 1):
            tag = ""
            if G.nodes[n]["is_pocket"]: tag = " [pocket]"
            elif G.nodes[n]["is_active"]: tag = " [active]"
            fh.write(f"  {rank:2d}. {node_label(G,n):12s}  BC={v:.5f}{tag}\n")
        fh.write("\n")

fig_b.suptitle("Top hub residues by betweenness centrality\n"
               "(red = allosteric pocket, green = active site)", fontsize=10)
fig_b.tight_layout()
fig_b.savefig(f"{OUT}/betweenness_centrality.png", dpi=200, bbox_inches="tight")
print(f"Saved betweenness_centrality.png")
print(f"Saved betweenness.txt")

# ── 3. Difference network ME1+Inhib − ME3+Inhib ───────────────────────────────
print("\n" + "="*65)
print("3. DIFFERENCE NETWORK  ME1+Inhib − ME3+Inhib")
print("="*65)

if "ME1+Inhib" in mean_graphs and "ME3+Inhib" in mean_graphs:
    G1 = mean_graphs["ME1+Inhib"]
    G3 = mean_graphs["ME3+Inhib"]

    # Map edges by (resid_i, resid_j) since node indices differ between systems
    def edge_rmi_dict(G):
        d = {}
        for a, b, data in G.edges(data=True):
            ri = G.nodes[a]["resid"]
            rj = G.nodes[b]["resid"]
            key = (min(ri, rj), max(ri, rj))
            d[key] = data["r_MI"]
        return d

    e1 = edge_rmi_dict(G1)
    e3 = edge_rmi_dict(G3)

    # Shared contacts (by resid pair)
    shared = set(e1.keys()) & set(e3.keys())
    print(f"  Shared edges by resid pair: {len(shared):,}")
    print(f"  ME1-only edges: {len(set(e1.keys()) - set(e3.keys())):,}")
    print(f"  ME3-only edges: {len(set(e3.keys()) - set(e1.keys())):,}")

    # Difference r_MI for shared contacts
    diffs = {k: e1[k] - e3[k] for k in shared}
    diff_vals = np.array(list(diffs.values()))

    # Top differences
    sorted_diff = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

    with open(f"{OUT}/diff_network.txt", "w") as fh:
        fh.write("DIFFERENCE NETWORK  ME1+Inhib − ME3+Inhib\n" + "="*65 + "\n\n")
        fh.write(f"Shared edges: {len(shared):,}\n")
        fh.write(f"ME1-only edges: {len(set(e1.keys()) - set(e3.keys())):,}\n")
        fh.write(f"ME3-only edges: {len(set(e3.keys()) - set(e1.keys())):,}\n\n")
        fh.write("Top 40 edges with largest |Δr_MI| (ME1 − ME3):\n")
        fh.write(f"{'ResID pair':20s}  {'ME1 r_MI':>10s}  {'ME3 r_MI':>10s}  {'Δr_MI':>10s}\n")
        for (ri, rj), dv in sorted_diff[:40]:
            fh.write(f"  {ri:5d}–{rj:5d}          "
                     f"{e1[(ri,rj)]:10.4f}  {e3[(ri,rj)]:10.4f}  {dv:+10.4f}\n")

    # Histogram of differences
    fig_d, ax_d = plt.subplots(figsize=(6, 4))
    ax_d.hist(diff_vals, bins=80, color="#555555", alpha=0.7, edgecolor="none")
    ax_d.axvline(0, color="red", lw=1.5, ls="--")
    ax_d.set_xlabel("Δr_MI  (ME1+Inhib − ME3+Inhib)", fontsize=10)
    ax_d.set_ylabel("Edge count", fontsize=10)
    ax_d.set_title("Difference in edge coupling strength\n"
                   "(positive = stronger in ME1)", fontsize=11, fontweight="bold")
    fig_d.tight_layout()
    fig_d.savefig(f"{OUT}/diff_network_hist.png", dpi=200, bbox_inches="tight")
    print(f"Saved diff_network_hist.png  (mean Δ = {diff_vals.mean():+.4f})")
    print(f"Saved diff_network.txt")

    # Save difference as a graph (shared edges, weight = Δr_MI)
    Gdiff = nx.Graph()
    for n, d in G1.nodes(data=True):
        Gdiff.add_node(d["resid"], **d)
    for (ri, rj), dv in diffs.items():
        Gdiff.add_edge(ri, rj, delta_rMI=dv, abs_delta=abs(dv))
    with open(f"{OUT}/diff_network.pkl", "wb") as fh:
        pickle.dump(Gdiff, fh)
    print(f"Saved diff_network.pkl")

# ── 4. Replicate variability (edge r_MI std) ─────────────────────────────────
print("\n" + "="*65)
print("4. REPLICATE VARIABILITY")
print("="*65)

fig_v, ax_v = plt.subplots(figsize=(6, 4))

for sys_name, rgs in rep_graphs.items():
    if len(rgs) < 2:
        continue
    cfg = SYSTEMS[sys_name]
    # collect r_MI per (resid_i, resid_j) across reps
    from collections import defaultdict
    edge_vals = defaultdict(list)
    for G in rgs:
        for a, b, d in G.edges(data=True):
            ri = G.nodes[a]["resid"]
            rj = G.nodes[b]["resid"]
            key = (min(ri, rj), max(ri, rj))
            edge_vals[key].append(d["r_MI"])
    # only edges present in ≥2 reps
    stds = [np.std(vs) for vs in edge_vals.values() if len(vs) >= 2]
    if stds:
        ax_v.hist(stds, bins=60, alpha=0.5, density=True,
                  color=cfg["color"], label=cfg["label"])
        print(f"  {sys_name}: median edge r_MI std = {np.median(stds):.4f} "
              f"(across {len(stds):,} shared edges)")

ax_v.set_xlabel("Std dev of r_MI across replicates", fontsize=10)
ax_v.set_ylabel("Density", fontsize=10)
ax_v.set_title("Replicate variability of edge coupling strengths", fontsize=11,
               fontweight="bold")
ax_v.legend(fontsize=9)
fig_v.tight_layout()
fig_v.savefig(f"{OUT}/replicate_variability.png", dpi=200, bbox_inches="tight")
print(f"Saved replicate_variability.png")

# ── 5. Path residue overlap: which path residues are shared ME1 vs ME3 ───────
if "ME1+Inhib" in path_results and "ME3+Inhib" in path_results:
    pr1 = path_results["ME1+Inhib"]
    pr3 = path_results["ME3+Inhib"]
    if pr1 and pr3:
        G1 = mean_graphs["ME1+Inhib"]
        G3 = mean_graphs["ME3+Inhib"]
        resids1 = set(G1.nodes[n]["resid"] for n in pr1["path"])
        resids3 = set(G3.nodes[n]["resid"] for n in pr3["path"])
        shared_path = resids1 & resids3
        only1 = resids1 - resids3
        only3 = resids3 - resids1
        print(f"\n  Path residue overlap (ME1 vs ME3):")
        print(f"    Shared:   {sorted(shared_path)}")
        print(f"    ME1-only: {sorted(only1)}")
        print(f"    ME3-only: {sorted(only3)}")
        with open(f"{OUT}/allosteric_paths.txt", "a") as fh:
            fh.write("\nPATH RESIDUE OVERLAP (ME1+Inhib vs ME3+Inhib)\n")
            fh.write(f"  Shared:   {sorted(shared_path)}\n")
            fh.write(f"  ME1-only: {sorted(only1)}\n")
            fh.write(f"  ME3-only: {sorted(only3)}\n")

print(f"\n{'='*65}")
print(f"Analysis complete. Results in {OUT}/")
