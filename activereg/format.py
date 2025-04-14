
from pathlib import Path

# Root of the git repo
REPO_ROOT = Path(__file__).resolve().parents[1]

# dir for the examples
EXAMPLES_REPO = REPO_ROOT / 'examples'

# file formats
FILE_TO_VAL = 'X_next_cycle_{}.csv'
FILE_VALIDATED = 'X_next_cycle_{}_validated.csv'