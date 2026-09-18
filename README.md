# FBA-guided decision-support reproducibility package

This is the single public package for the JSET manuscript. It contains anonymous
rating data, analysis code, generated results, the sanitized agent workflow, and
the ethics approval certificate.

The research system structures teacher-provided information about student problem
behavior through an FBA-guided workflow. The evidence in this package covers
teacher perceptions and expert ratings. Diagnostic accuracy, intervention effects,
student outcomes, and high-stakes professional decisions require separate studies.

## Package contents

| Path | Contents |
| --- | --- |
| `data/public_ratings/` | Anonymous teacher and expert six-dimension rating files. |
| `data/public_metadata/` | Data dictionary, sample flow, expert-round metadata, and analysis reconciliation notes. |
| `code/` | Input validation, analysis, plotting, and supplementary-table code. |
| `results/` | Generated aggregate tables, figures, and run metadata. |
| `workflow/` | Sanitized public description of the agent workflow. |
| `docs/ethics/` | Ethics approval certificate, approval no. `BNU202601050010`. |
| `docs/` | Deidentification, provenance, release, and analysis notes. |

## Public rating data

`teacher_ratings_deidentified.csv` contains one row for each of 137 teachers.
Every teacher received a new identifier from `P001` through `P137`. The file keeps
only the identifier and three six-dimension rating blocks: report quality,
similarity to practitioner-conducted FBA, and similarity to a general-purpose AI.
Blank blocks indicate that the corresponding evaluation was outside that
teacher's questionnaire.

The practitioner-referenced analysis contains 75 distinct teachers with complete
six-dimension practitioner-FBA similarity ratings. The matched comparison contains
62 teachers with complete six-dimension general-purpose-AI similarity ratings.
Among all 137 teachers, 121 supplied complete six-dimension report-quality ratings.

The expert files use `E001` through `E004`. The paired-round file contains one
Round 1 and one Round 2 record for each expert. The final expert file contains the
arithmetic mean of the two rounds, with recorded decimal values preserved.

Collection mode, original identifiers, names, teaching stage, grade, contact
information, location/device metadata, screenshots, URLs, narratives, and raw chat
content are absent from the public rating files.

## Reproduce the results

Use Python 3.10 or later.

```powershell
python -m pip install -r code\requirements.txt
python code\run_submission_analysis.py --overwrite
```

The analysis validates the public inputs, regenerates the result tables and figure
files, and rebuilds the six consolidated supplementary tables. The code performs
schema, range, pairing, arithmetic-average, and sample-count checks.

## Workflow and ethics files

`workflow/functional_behavior_assessment_workflow_public.yaml` preserves the node
structure, input/output schemas, prompt logic, code blocks, and high-level edges.
Deployment identifiers, credentials, signed links, and private retrieval content
have been replaced or omitted.

The agent's model components were updated after the study data were collected.
The uploaded sanitized workflow is derived from the latest agent version rather
than the historical study-time deployment; the manuscript separately reports the
models documented for the data-collection period. Model endpoints, identifiers,
and credentials remain replaced with placeholders in the public file.

The retrieval knowledge base is not uploaded. It is linked to specific case
materials collected for this study and is withheld to protect participant privacy
and system security.

`docs/ethics/ethics_approval_BNU202601050010.pdf` is the study's three-page ethics
approval certificate. Its public copy retains the page content and approval number;
document metadata has been reduced to a public title and subject.
