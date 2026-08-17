"""Backfill SHA-256 integrity hashes into models/registry.json.

`AnomalyDetector` refuses to deserialise a model whose registry entry carries
no `model_hash` / `scaler_hash` (joblib.load unpickles, which executes code).
Registries written before those fields existed therefore need a one-off
backfill; `src/train.py` records them automatically for new models.

The script is idempotent: entries that already carry a matching hash are left
untouched, and it only rewrites the registry when something actually changed.
Entries whose `.joblib` files are absent are skipped and reported, so the same
command can be re-run on the machine that holds the artefacts.

Usage:
    python scripts/backfill_model_hashes.py
    python scripts/backfill_model_hashes.py --models-dir models --dry-run
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HASH_FIELDS = (("model_file", "model_hash"), ("scaler_file", "scaler_hash"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backfill_registry(models_dir: str | Path, dry_run: bool = False) -> dict:
    """Compute and record missing hashes for every entry in the registry.

    Args:
        models_dir: Directory holding registry.json and the .joblib artefacts.
        dry_run: If True, report what would change without writing.

    Returns:
        Dict with 'updated' (list of "version:field"), 'unchanged' (same shape),
        'missing_files' (list of filenames not found), and 'mismatched' (list of
        "version:field" whose recorded hash disagrees with the file on disk).

    Raises:
        FileNotFoundError: If the registry does not exist.
    """
    models_path = Path(models_dir)
    registry_path = models_path / "registry.json"

    if not registry_path.is_file():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path) as f:
        registry = json.load(f)

    updated: list[str] = []
    unchanged: list[str] = []
    missing_files: list[str] = []
    mismatched: list[str] = []

    for entry in registry.get("models", []):
        version = entry.get("version", "?")

        for file_field, hash_field in HASH_FIELDS:
            filename = entry.get(file_field)
            if not filename:
                continue

            artefact = models_path / filename
            if not artefact.is_file():
                missing_files.append(filename)
                continue

            actual = sha256_file(artefact)
            recorded = entry.get(hash_field)

            if recorded == actual:
                unchanged.append(f"{version}:{hash_field}")
            elif recorded:
                mismatched.append(f"{version}:{hash_field}")
            else:
                entry[hash_field] = actual
                updated.append(f"{version}:{hash_field}")

    if updated and not dry_run:
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    return {
        "updated": updated,
        "unchanged": unchanged,
        "missing_files": missing_files,
        "mismatched": mismatched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing registry.json and the .joblib files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing the registry",
    )
    args = parser.parse_args()

    result = backfill_registry(args.models_dir, dry_run=args.dry_run)

    for item in result["updated"]:
        print(f"  recorded hash: {item}")
    for item in result["unchanged"]:
        print(f"  already correct: {item}")
    for filename in result["missing_files"]:
        print(f"  skipped (file not found): {filename}")
    for item in result["mismatched"]:
        print(f"  MISMATCH — file on disk differs from the recorded hash: {item}")

    if result["mismatched"]:
        print(
            "\nRefusing to overwrite a recorded hash. Either the artefact was "
            "replaced or the registry is wrong — resolve this by hand.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run and result["updated"]:
        print("\nDry run — registry not written.")
    elif result["updated"]:
        print(f"\nRegistry updated: {len(result['updated'])} hash(es) recorded.")
    else:
        print("\nNothing to do — every present artefact already has a hash.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
