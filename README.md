# Business Entity Resolution Challenge

Entity resolution pipeline for matching Source 1 business records
against corresponding records from Source 2 and Source 3.

## Project Structure

```text
dataset/
├── train/
└── test/

src/
└── main.py

utils/
└── validate_submission.py

output/
├── matching_results.tsv
└── candidate_pairs.tsv