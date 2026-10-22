# SpectroLung S2S deliverables

This directory contains the reproducible implementation and generated reports for
the HF_Lung inhale/exhale span task. Original data and annotations are never
modified. Commands are run from the repository root:

```powershell
python deliverables\scripts\run_inventory.py --data-root HF_Lung
python deliverables\scripts\run_roundtrip.py --annotations HF_Lung\annotations.csv
```

The scripts write generated artifacts below `deliverables\results`. Paths to
large audio and DVC-managed data remain external; the manifests record the
source paths and configuration used.
