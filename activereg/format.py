
from pathlib import Path

# Root of the git repo
REPO_ROOT = Path(__file__).resolve().parents[1]

# Dir definitnions
DATASETS_REPO = REPO_ROOT / 'datasets'
EXAMPLES_REPO = REPO_ROOT / 'examples'
BENCHMARKS_REPO = REPO_ROOT / 'benchmarks'
INSILICO_AL_REPO = REPO_ROOT / 'insilico_al'
EXPERIMENTS_REPO = REPO_ROOT / 'experiments'
LAB_AL_REPO = REPO_ROOT / 'lab_al_experiments'


def resolve_dataset_path(file_name: str, root: Path = None) -> Path:
    """Resolve a dataset file name, transparently accepting a gzipped variant.

    Large ground-truth landscapes are stored gzipped so they stay under the
    100 MB per-file limit of common git hosts.  ``pandas.read_csv`` decompresses
    ``.csv.gz`` transparently, so only the path lookup needs to be aware of it.
    Config files therefore keep naming the plain ``.csv`` — they are provenance
    records of what actually ran and are not rewritten.

    The lookup is symmetric: a name given with or without the ``.gz`` suffix
    resolves to whichever variant is present on disk.

    Args:
        file_name (str): Dataset file name, with or without a ``.gz`` suffix.
        root (Path, optional): Directory to resolve against. Defaults to
            ``DATASETS_REPO``.

    Returns:
        Path: Path to the file that exists on disk.

    Raises:
        FileNotFoundError: If neither the plain nor the gzipped variant exists.
    """
    root = DATASETS_REPO if root is None else Path(root)
    name = str(file_name)

    candidates = [name[:-3], name] if name.endswith('.gz') else [name, name + '.gz']
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"dataset '{name}' not found in {root} (tried: {', '.join(candidates)})"
    )

