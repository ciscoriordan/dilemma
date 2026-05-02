"""Download Dilemma data files from HuggingFace Hub.

    python -m dilemma download                  # everything (lemma + tagger weights)
    python -m dilemma download --dir /some/path
    python -m dilemma download --no-tagger      # lemma data only
    python -m dilemma download --only-tagger    # tagger weights only

Both lemma data and tagger weights live at
https://huggingface.co/ciscoriordan/dilemma (~2.7 GB combined). The tagger
weights are under the `tagger/` prefix and the lemma artifacts are under
`data/` and `model/`.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "dilemma"
REPO = "ciscoriordan/dilemma"
INCLUDES = ["data/*", "model/*"]

TAGGER_REPO = "ciscoriordan/dilemma"
TAGGER_INCLUDES = ["tagger/*"]


def _snapshot_download(*, repo_id, local_dir, allow_patterns):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download Dilemma data. "
            "Install with: pip install huggingface_hub"
        ) from e
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )


def download(target_dir: Path | None = None, *, tagger: bool = True,
             dilemma: bool = True) -> Path:
    """Download Dilemma + tagger artifacts from HuggingFace to `target_dir`.

    Lemma `data/` and `model/` go directly under `target_dir`. Tagger
    weights land at `target_dir/tagger_model/` (mirrors `~/.cache/dilemma/`
    layout: `model/` for the lemmatizer, `tagger_model/` for the tagger).

    Returns the path that was downloaded into. Requires `huggingface_hub`.
    """
    dest = Path(target_dir) if target_dir else DEFAULT_CACHE
    dest.mkdir(parents=True, exist_ok=True)

    if dilemma:
        _snapshot_download(
            repo_id=REPO, local_dir=dest, allow_patterns=INCLUDES,
        )
    if tagger:
        tagger_dest = dest / "tagger_model"
        tagger_dest.mkdir(parents=True, exist_ok=True)
        # The HF repo lays out tagger weights under `tagger/<lang>/...`. We
        # strip the leading `tagger/` so files land at
        # `tagger_model/<lang>/...`, which is what
        # dilemma.tagger._WEIGHTS_DIR expects.
        _snapshot_download(
            repo_id=TAGGER_REPO,
            local_dir=tagger_dest.parent / "_tagger_tmp",
            allow_patterns=TAGGER_INCLUDES,
        )
        _flatten_tagger_weights(tagger_dest.parent / "_tagger_tmp", tagger_dest)

    return dest


def _flatten_tagger_weights(src: Path, dst: Path) -> None:
    """Move src/tagger/<lang>/... -> dst/<lang>/...; remove src tree."""
    import shutil
    weights_root = src / "tagger"
    if not weights_root.exists():
        return
    for child in weights_root.iterdir():
        target = dst / child.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(child), str(target))
    shutil.rmtree(src, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dilemma download",
        description="Download Dilemma lookup tables, lemma model, and tagger weights.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Target directory (default: {DEFAULT_CACHE})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-tagger", action="store_true",
        help="Skip tagger POS / dep weights (lemma data only)",
    )
    group.add_argument(
        "--only-tagger", action="store_true",
        help="Download only tagger weights (skip lemma data + model)",
    )
    args = parser.parse_args(argv)
    dest = download(
        args.dir,
        tagger=not args.no_tagger,
        dilemma=not args.only_tagger,
    )
    print(f"Downloaded to {dest}")
    print(
        "Dilemma will find this automatically, or set "
        f"DILEMMA_DATA_DIR={dest / 'data'} to override."
    )
    if not args.no_tagger:
        print(
            "Tagger weights at "
            f"{dest / 'tagger_model'}; override with DILEMMA_TAGGER_DIR."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
