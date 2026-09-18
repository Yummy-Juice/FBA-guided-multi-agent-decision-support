# Analysis code

This directory contains the public validation, analysis, plotting, and
supplementary-table workflow.

```powershell
python -m pip install -r code\requirements.txt
python code\run_submission_analysis.py --overwrite
```

`validate_inputs.py` checks the 137 anonymous teacher records, 75-teacher
practitioner-referenced cohort, 62-teacher matched comparison, four expert final
records, and eight paired expert-round records. It validates schema, identifiers,
rating ranges, complete six-item blocks, cohort counts, round pairing, and the
arithmetic-average final expert scores.

`run_submission_analysis.py` validates the inputs and writes the tables and
figures under `../results/`. Each figure is exported as PNG, TIFF, PDF, and SVG.
The same command calls `build_consolidated_supplement.py` after the primary
analysis and rebuilds the six public supplementary CSV tables.

The code analyzes perception ratings and expert ratings. The resulting evidence
supports the reported descriptive, reliability, inferential, and sensitivity
analyses within those rating designs.
