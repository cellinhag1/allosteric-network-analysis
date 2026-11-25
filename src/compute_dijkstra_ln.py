"""
Step 2: Dijkstra + Yen's K-shortest paths using -ln(c) weight (literature standard).
Parallelized by source residue index via SLURM array.

Usage: python compute_dijkstra_ln.py <src_idx>
  src_idx: index into SRC_RESIDS list (0-13)

Saves: data_cog/dijkstra_ln_src{src_resid}.npy
"""
import os, sys
import numpy as np
import networkx as nx
from scipy import stats

src_idx = int(sys.argv[1])

ALLOD = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(ALLOD, 'data_cog')
w     = 'last25ns'

SRC_RESIDS = [259,260,263,280,281,284,314,317,321,322,325,360,485,488]
DST_RESIDS = [112,165,183,418,463]
src_res    = SRC_RESIDS[src_idx]

resids = np.load(os.path.join(DATA, 'me1_inh_resids.npy'))
lu     = {int(r): i for i, r in enumerate(resids)}
N      = len(resids)
SEQ_SEP = 10
K_PATHS = 5   # number of suboptimal paths per source-dest pair

if src_res not in lu:
    print(f'Residue {src_res} not in trajectory — exiting'); sys.exit(0)

mi_h  = np.load(os.path.join(DATA, f'me1_inh_{w}_mi_mean.npy')).astype(np.float64)
mi_a  = np.load(os.path.join(DATA, f'me1_noinh_{w}_mi_mean.npy')).astype(np.float64)
dmi   = mi_h - mi_a

# ── Merge contact map from per-rep files ─────────────────────────────────────
total_counts  = np.zeros((N,N), dtype=np.int64)
total_frames  = 0
for rep in range(25):
    fc = os.path.join(DATA, f'contact_counts_rep{rep}.npy')
    fn = os.path.join(DATA, f'contact_nframes_rep{rep}.npy')
    if os.path.isfile(fc) and os.path.isfile(fn):
        total_counts  += np.load(fc).astype(np.int64)
        total_frames  += int(np.load(fn)[0])

if total_frames == 0:
    print('ERROR: no contact map files found. Run compute_contact_map_cog first.')
    sys.exit(1)

contact_frac = total_counts / total_frames
contact      = contact_frac >= 0.20    # ≥20% of frames
np.fill_diagonal(contact, False)
print(f'Contact pairs (≥20% frames): {contact.sum()//2}', flush=True)

# ── FDR significant pairs ────────────────────────────────────────────────────
def bh_reject(pv, alpha=0.05):
    n=len(pv); order=np.argsort(pv); rp=pv[order]
    below=rp<=alpha*np.arange(1,n+1)/n
    if not below.any(): return np.zeros(n,dtype=bool)
    mk=np.where(below)[0].max()
    ro=np.zeros(n,dtype=bool); ro[:mk+1]=True
    r=np.empty(n,dtype=bool); r[order]=ro; return r

ai=np.load(os.path.join(DATA,f'me1_inh_{w}_mi_all.npy')).astype(np.float64)
an=np.load(os.path.join(DATA,f'me1_noinh_{w}_mi_all.npy')).astype(np.float64)
tr,tc=np.triu_indices(N,k=1)
_,pv=stats.ttest_ind(ai[:,tr,tc],an[:,tr,tc],axis=0,equal_var=False)
rej=bh_reject(pv,0.05)
sig_set={(min(int(tr[k]),int(tc[k])),max(int(tr[k]),int(tc[k]))) for k in np.where(rej)[0]}
sig_resids={int(resids[tr[k]]) for k in np.where(rej)[0]}|{int(resids[tc[k]]) for k in np.where(rej)[0]}

# ── Build graph with -ln(rMI) weights ───────────────────────────────────────
eps     = 1e-6
w_lnmi  = -np.log(np.clip(mi_h, eps, 1-eps))

G = nx.Graph(); G.add_nodes_from(range(N))
rows,cols = np.where(np.triu(contact, k=SEQ_SEP))
for i,j in zip(rows,cols):
    G.add_edge(int(i),int(j), weight=float(w_lnmi[i,j]))
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges', flush=True)

# ── Yen's K-shortest from this source to all destinations ───────────────────
s = lu[src_res]
node_freq   = {}   # relay residue → count of paths it appears in
all_paths   = []   # (weight, path_residues, dst_residue)

for dst_res in DST_RESIDS:
    if dst_res not in lu: continue
    d = lu[dst_res]
    try:
        k_paths = list(nx.shortest_simple_paths(G, s, d, weight='weight'))[:K_PATHS]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        continue
    for path in k_paths:
        w_total = nx.path_weight(G, path, 'weight')
        path_res = [int(resids[i]) for i in path]
        all_paths.append((w_total, path_res, dst_res))
        for node in path[1:-1]:   # relay residues only (exclude src/dst)
            r = int(resids[node])
            node_freq[r] = node_freq.get(r,0) + 1

all_paths.sort()

# ── Print results ────────────────────────────────────────────────────────────
print(f'\nSource: {src_res}  |  {len(all_paths)} paths found across {len(DST_RESIDS)} destinations')
print(f'\nTop 5 paths:')
for rank,(wtot,path_res,dst) in enumerate(all_paths[:5]):
    hops  = len(path_res)-1
    relays= path_res[1:-1]
    n_sig = sum(1 for r in relays if r in sig_resids)
    dvals = [float(dmi[lu[path_res[k]],lu[path_res[k+1]]]) for k in range(len(path_res)-1)]
    print(f'  Rank {rank+1}: {" → ".join(map(str,path_res))}')
    print(f'    dst={dst}  hops={hops}  weight={wtot:.4f}'
          f'  sig_relays={n_sig}/{len(relays)}  mean_dmi={np.mean(dvals):+.4f}')
    for k in range(len(path_res)-1):
        ra,rb=path_res[k],path_res[k+1]
        ia,ib=lu[ra],lu[rb]
        sig='*FDR*' if (min(ia,ib),max(ia,ib)) in sig_set else ''
        s_tag='[SIG]' if ra in sig_resids else ''
        print(f'      {ra}{s_tag} ↔ {rb}{"[SIG]" if rb in sig_resids else ""}'
              f'  rMI={mi_h[ia,ib]:.4f}  ΔMI={dmi[ia,ib]:+.4f}  {sig}')

print(f'\nTop 15 recurring relay residues across {K_PATHS} suboptimal paths:')
top_recurring=sorted(node_freq.items(),key=lambda x:-x[1])[:15]
print(f'  {"Resid":>6}  {"count":>6}  {"FDR sig?":>10}  {"mean ΔMI":>10}')
for r,cnt in top_recurring:
    is_sig='*** YES' if r in sig_resids else 'no'
    idx=lu.get(r)
    dmean=float(dmi[idx].mean()) if idx is not None else 0
    print(f'  {r:>6d}  {cnt:>6d}  {is_sig:>10s}  {dmean:>+10.4f}')

# Save results
out = os.path.join(DATA, f'dijkstra_ln_src{src_res}.npz')
np.savez(out,
         top_paths=np.array([(w,p[:10]) for w,p,_ in all_paths[:20]], dtype=object),
         node_freq_keys=np.array(list(node_freq.keys())),
         node_freq_vals=np.array(list(node_freq.values())))
print(f'\nSaved: {out}', flush=True)
