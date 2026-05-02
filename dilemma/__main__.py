"""`python -m dilemma <subcommand>` dispatcher."""

import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m dilemma <command> [args...]")
        print("")
        print("Commands:")
        print("  download   Download lookup tables and model files from HuggingFace")
        print("  paradigm   Generate / fill Greek inflection paradigms")
        return 0
    cmd, *rest = sys.argv[1:]
    if cmd == "download":
        from ._download import main as download_main
        return download_main(rest)
    if cmd == "paradigm":
        from .paradigm import _cli as paradigm_cli
        return paradigm_cli(rest)
    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
