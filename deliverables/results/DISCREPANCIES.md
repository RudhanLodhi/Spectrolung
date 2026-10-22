# Discrepancies and limitations

- The repository contains an empty `deliverable` directory, while this package
  is intentionally placed in the requested `deliverables` directory.
- Existing notebooks are not treated as executable source of truth. Their
  preprocessing and model settings must be compared against the logged config
  before any model result is reported.
- HF_Lung phase labels are limited to `I` and `E`; other labels are retained in
  the source dataset but mapped to background for this span target.
- No clinical-validity claim is made. Any proposed annotation correction must be
  versioned and reviewed separately.
