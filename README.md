# allosteric-network-analysis

Tools for finding the residue network that carries an allosteric signal from a
ligand-binding pocket to a distant active site, using an **ensemble** of
independent molecular dynamics trajectories rather than one long run.

The question these answer: when a ligand binds an allosteric pocket, which
residues actually relay that information to the active site, and does the relay
change between two isoforms of the same enzyme?

## Why an ensemble

Allosteric coupling is measured from *correlated motion*, and correlation is a
statistic. A single long trajectory can settle into one basin and report
correlations that are real for that basin and wrong for the equilibrium
ensemble. Running many independent replicates with different initial velocities
and averaging the correlation matrices suppresses that noise and lets you put an
error bar on every residue pair.

The scripts assume this layout: several conditions (for example apo and
holo, for two isoforms), each with `N_REPS` independent replicates.

## Pipeline

Run in order; each stage writes arrays the next one reads.

**1. Reduce the trajectories**

| Script | Purpose |
|---|---|
| `strip_water_mda.py` | Drop solvent, keep protein and ligands |
| `fix_imaging_inhibitor.py` | Undo periodic imaging so the ligand stays whole and next to its pocket |
| `extract_ca_coords.py` | Cα coordinates per replicate |
| `extract_cog_coords.py` | Per-residue centre-of-geometry coordinates (sensitive to side-chain motion, unlike Cα) |

**2. Correlation and the network**

| Script | Purpose |
|---|---|
| `compute_dccm_mi.py`, `compute_dccm_mi_cog.py` | Dynamic cross-correlation (DCCM) and generalised correlation from Gaussian mutual information, per replicate, then averaged |
| `compute_contact_map_cog.py` | Contact persistence, used to keep only physically adjacent pairs as graph edges |
| `network_build.py` | Build weighted `NetworkX` graphs: nodes are residues, edge weights are generalised correlation on persistent contacts |
| `network_analyze.py` | Betweenness centrality and hub detection on those graphs |
| `compute_dijkstra_ln.py` | Shortest communication path from pocket to active site, with edge length `-ln(correlation)` so that strongly correlated pairs are *short* |

**3. Conformational change and statistics**

| Script | Purpose |
|---|---|
| `compute_openclosed_metrics.py`, `summarize_openclosed_metrics.py` | Catalytic distances and cleft geometry that distinguish open from closed |
| `compute_domain_rotation.py`, `compute_hinge_axis.py` | Rigid-body domain rotation and the hinge it turns about |
| `compute_ca_pca.py` | Principal components of the Cα fluctuations |
| `compute_rmsf.py`, `compute_rmsf_ttest.py` | Per-residue fluctuation, with a Welch t-test across replicates |
| `stats_allostery.py`, `stats_threeway.py` | Condition comparisons with Benjamini-Hochberg FDR control |

**4. Ligand stability**

| Script | Purpose |
|---|---|
| `compute_sasa_inhibitor.py` | Shrake-Rupley solvent-accessible surface of the ligand, i.e. how buried it stays |
| `plot_stability.py` | Pocket centre-of-mass distance and SASA over time, one condition against another |

## The `-ln(correlation)` trick

Dijkstra finds the path of least total *length*, but the interesting path is the
one of greatest total *correlation*. Setting each edge's length to
`-ln(r_MI)` converts the product of correlations along a path into a sum of
lengths, so the shortest path is exactly the most strongly correlated chain of
residues between the two endpoints. Correlations near 1 give lengths near 0;
weak correlations become very long edges and are avoided.

## Configuration

No path or system name is hard-coded. `src/config.py` resolves everything and
each setting can be overridden from the environment:

```bash
export ALLOSTERY_DATA=/scratch/my_runs   # root of the trajectory tree
export N_REPS=25                         # replicates per condition
export LIGAND_RESNAME=LIG                # the bound small molecule
export POCKET_CUTOFF=8.0                 # Angstrom, ligand-in-pocket test
python compute_dccm_mi_cog.py holo 5000  # condition, first frame to use
```

The second argument to the correlation scripts is the first frame to keep, so
early unequilibrated blocks can be discarded without recomputing anything else.

## Requirements

Python 3, NumPy, SciPy, NetworkX, MDAnalysis, biotite, Matplotlib. The imaging
fix uses pyCHARMM and so needs a CHARMM installation; everything downstream
works from the saved coordinate arrays and does not.

## License

MIT, see `LICENSE`.
