# Anonymous public rating data

This directory contains the individual-level rating matrices used by the public
analysis code.

| File | Rows | Identifier | Contents |
| --- | ---: | --- | --- |
| `teacher_ratings_deidentified.csv` | 137 | `P001` through `P137` | Anonymous teacher identifiers and three six-dimension rating blocks. |
| `expert_delphi_rounds_deidentified.csv` | 8 | `E001` through `E004` | Two paired expert rounds with six ratings per record. |
| `expert_ratings_deidentified.csv` | 4 | `E001` through `E004` | Arithmetic mean of each expert's paired Round 1 and Round 2 ratings. |

The teacher table excludes collection mode and all source identity or demographic
fields. Its blank cells preserve the questionnaire structure. The expert overall
rating keeps the decimal precision recorded in the source data.

All files use UTF-8, one record per row, and English snake_case field names. See
`../public_metadata/data_dictionary.csv` for definitions and value ranges.
