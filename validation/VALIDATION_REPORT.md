# Validation report

## Public package run

The public package completed one full reproduction run on 2026-09-17 local time
(2026-09-18 UTC). The entry point was:

```powershell
python -B code\run_submission_analysis.py --overwrite
```

The same run validated the public inputs, regenerated the primary analysis
outputs and four figure sets, and rebuilt all six consolidated supplementary
tables.

## Public input checks

- Teacher file: 137 rows and 19 columns, with unique identifiers `P001` through
  `P137` and rating fields only.
- Teacher report quality: 121 complete six-dimension records.
- Practitioner-referenced similarity: 75 complete six-dimension records from 75
  distinct teachers.
- Matched general-purpose-AI comparison: 62 complete six-dimension records.
- Expert final file: four records, `E001` through `E004`.
- Expert paired-round file: eight records, with one Round 1 and one Round 2 row
  for each expert.
- Every final expert value equals the arithmetic mean of the matching paired
  round values.
- Every rating value is numeric and within the documented 1-5 range; expert
  decimals are preserved.

## Table S3 check

The expert Overall rating distribution in consolidated Table S3 accounts for all
four final expert records:

| Rating | n | Proportion |
| ---: | ---: | ---: |
| 3.0 | 2 | 0.50 |
| 3.5 | 1 | 0.25 |
| 4.2 | 1 | 0.25 |

The total is `n=4` and the proportions sum to `1.00`. The two ratings outside the
integer value 3 are retained at their recorded decimal precision.

## Generated outputs

- 31 primary result CSV files across the three evaluation directories;
- six consolidated supplementary CSV tables;
- four figures in PNG, TIFF, PDF, and SVG formats, for 16 figure files;
- one run manifest with the public input filenames, sample counts, methods, and
  generated figure names.

## Privacy and release checks

- The teacher file contains a new anonymous number and 18 rating fields.
- Collection mode, names, source IDs, teaching stage, grade, contact information,
  location/device metadata, screenshots, URLs, free text, narratives, and chat
  content are absent from the public data.
- The public metadata use the same privacy boundary and contain no collection-mode
  breakdown.
- The sanitized workflow contains placeholders in place of deployment identifiers,
  credentials, signed links, and private retrieval content.
- The ethics certificate has three pages, contains approval no.
  `BNU202601050010`, and uses public PDF metadata without an author field.

The project instructions exclude visual QA and independent digest verification;
this report records structural, data, privacy, and reproducibility checks.
