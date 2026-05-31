from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data import build_index
except ImportError:
    from data import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    index = build_index(args.data_root, skip_missing_files=True)
    print(f"Valid samples: {len(index)}")
    print(f"Skipped missing files: {index.attrs.get('skipped_missing_files', 0)}")
    print("\nBy split:")
    print(index["split_name"].value_counts())
    print("\nBy label:")
    print(index["label"].value_counts())


if __name__ == "__main__":
    main()
