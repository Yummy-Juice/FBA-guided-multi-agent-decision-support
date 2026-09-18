# Deidentification log

## Teacher ratings

The public teacher table contains 137 records. The release process:

1. removed every original identity field and source record label;
2. assigned new sequential identifiers `P001` through `P137`;
3. retained only the three six-dimension numeric rating blocks;
4. removed collection mode, teaching stage, reported grade, questionnaire source
   metadata, contact details, device/location data, screenshot links, and text;
5. preserved missing values and recorded rating precision;
6. retained all 75 distinct teachers in the practitioner-referenced evaluation
   and all 62 teachers in the matched comparison.

## Expert ratings

The public expert files use `E001` through `E004`. Each expert has one record in
Round 1 and one in Round 2. The final-score file contains the arithmetic mean of
the paired rounds. Only the anonymous record number, round label, and six numeric
ratings are retained.

## Public ethics file

The three-page ethics approval certificate is included at
`ethics/ethics_approval_BNU202601050010.pdf`. The public copy keeps the certificate
pages and approval number and uses public document metadata without an author,
email, phone number, or identity number.

## Materials outside the release

The public package excludes original workbooks, names, source IDs, free-text
responses, student narratives, screenshots, raw chat logs, private application
exports, credentials, signed links, and the private retrieval corpus.
