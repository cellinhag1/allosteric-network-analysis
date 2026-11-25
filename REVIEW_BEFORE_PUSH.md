# Review before making public

All machine paths, colleague paths and the internal compound code name have
been removed. Three lines still carry system-specific residue numbers for the
allosteric pocket:

    src/fix_imaging_inhibitor.py:53    POCKET_RESID = [260, 263, 321, 360, 485]
    src/fix_imaging_inhibitor.py:62    POCKET_RESID = [285, 288, 346, 385, 510]
    src/compute_openclosed_metrics.py:77  POCKET_RESID = [260, 263, 321, 360, 485]

These say where the ligand binds. If the pocket is not yet in a published
paper, replace them with placeholders and document them as a required setting,
or confirm with your advisor that they are fine to publish.

Everything else in this repository is method code with no results in it.

Delete this file once you have decided.
