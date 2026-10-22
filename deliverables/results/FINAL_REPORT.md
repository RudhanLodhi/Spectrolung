# SpectroLung S2S final report

## Pipeline fixes

The deliverables package establishes a native 4 kHz, 100-frame-per-second
configuration, patient-key helpers, center-based frame labeling, and a
model-free temporal round-trip diagnostic. Padding and source annotations are
kept separate from generated artifacts.

## Model results

Model training and final metrics are intentionally not claimed until the
inventory, round-trip gate, cache verification, and patient-level validation
split have been run. Generated tables belong in this directory and must include
the configuration and seed used for every number.

## Label-quality findings

Only `I` and `E` are target phase labels. Other labels are not silently
relabelled. Audit exports must preserve source labels and proposed corrections
as separate versions.

## Open questions for Dr. Khan

Protocol differences around confirming pauses, artifacts, and unknown phase
remain review questions. Audio energy alone is not sufficient evidence for a
clinical phase decision.

## Limitations

This package does not establish clinical validity, prevalence, or annotation
quality from hand-picked clips. Results are reproducible only when the source
data, configuration, dependency versions, and seed are recorded together.
