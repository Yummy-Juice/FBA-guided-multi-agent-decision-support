# Code provenance

`code/run_submission_analysis.py` is the public entry point. It reads the
anonymous CSV files, validates the documented snapshot, writes the English result
tables, exports each figure as PNG, TIFF, PDF, and SVG, and rebuilds the six
consolidated supplementary tables.

`code/validate_inputs.py` validates the expected 137-teacher, four-expert-final,
and eight-row paired expert snapshot. Its checks cover identifier format, rating
schema and range, complete six-item blocks, the 75-teacher and 62-teacher cohort
counts, expert round pairing, and the arithmetic-average final expert scores.

The analysis reports complete ordinal distributions, median and IQR, Wilson
confidence intervals for P(rating >= 4), internal consistency, coefficient of
variation, multiplicity-adjusted tests, and ordinal sensitivity analyses. These
outputs describe the teacher-perception and expert-rating evidence collected in
the study.
