"""Download Dilemma data files from HuggingFace Hub.

    python -m dilemma download                 # default: ~/.cache/dilemma
    python -m dilemma download --dir /some/path

Data lives at https://huggingface.co/ciscoriordan/dilemma - about 1.6 GB.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "dilemma"
REPO = "ciscoriordan/dilemma"
INCLUDES = ["data/*", "model/*"]


def download(target_dir: Path | None = None) -> Path:
    """Download data + model files from HuggingFace to `target_dir`.

    Returns the path that was downloaded into. Requires `huggingface_hub`.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download Dilemma data. "
            "Install with: pip install huggingface_hub"
        ) from e

    dest = Path(target_dir) if target_dir else DEFAULT_CACHE
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        local_dir=str(dest),
        allow_patterns=INCLUDES,
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dilemma download",
        description="Download Dilemma lookup tables and model files.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Target directory (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args(argv)
    dest = download(args.dir)
    print(f"Downloaded to {dest}")
    print(
        "Dilemma will find this automatically, or set "
        f"DILEMMA_DATA_DIR={dest / 'data'} to override."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
