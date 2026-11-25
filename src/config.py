"""
Shared configuration for the allosteric network analysis scripts.

Every path and system-specific name used by the scripts in this directory is
resolved here, so nothing is hard-coded to one machine or one project.

Override any of these with an environment variable, for example:

    export ALLOSTERY_DATA=/scratch/me1_runs
    export LIGAND_RESNAME=LG00
    python compute_dccm_mi_cog.py me1_inh 5000
"""
import os

# Root of the trajectory / analysis tree. Scripts read and write beneath this.
BASE = os.environ.get("ALLOSTERY_DATA", os.path.abspath("."))

# Subdirectories, created on demand by the scripts that write to them.
ALLOD = os.path.join(BASE, "allostery")
DATA = os.path.join(ALLOD, "data_cog")
NETWORKS = os.path.join(BASE, "networks")
FIGURES = os.path.join(BASE, "figures")

# Ensemble size: number of independent replicates per condition.
N_REPS = int(os.environ.get("N_REPS", "25"))

# Ligand selection. The scripts refer to the bound small molecule through
# these, never by a hard-coded name.
LIGAND_RESNAME = os.environ.get("LIGAND_RESNAME", "LIG")
LIGAND_SEGID = os.environ.get("LIGAND_SEGID", "LIGA")
LIGAND_PARAMS = os.environ.get("LIGAND_PARAMS", "ligand.str")

# Geometric criteria.
POCKET_CUTOFF = float(os.environ.get("POCKET_CUTOFF", "8.0"))   # Angstrom
CONTACT_CUTOFF = float(os.environ.get("CONTACT_CUTOFF", "7.0"))  # Angstrom
CONTACT_PERSISTENCE = float(os.environ.get("CONTACT_PERSISTENCE", "0.25"))

# Force-field and trajectory locations.
TOPPAR = os.environ.get("TOPPAR", os.path.join(BASE, "toppar"))
CGENFF = os.environ.get("CGENFF", os.path.join(BASE, "cgenff"))
DYN = os.environ.get("DYN_DIR", os.path.join(BASE, "dynamics"))
OUT = os.environ.get("OUT_DIR", os.path.join(BASE, "analysis"))
