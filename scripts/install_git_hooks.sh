#!/bin/bash
# Install the repo's git hooks (they are not shared by a clone).
#
#   ./scripts/install_git_hooks.sh
#
# pre-push: refuses to push while the data artifacts CI downloads
# (data/hf_manifest.json) are out of sync with the Hub - either rebuilt
# locally and not uploaded, or uploaded but still in flight. Both make CI test
# new code against old data. Bypass with `git push --no-verify` or
# DILEMMA_SKIP_HF_CHECK=1.
set -e
cd "$(dirname "$0")/.."
HOOK="$(git rev-parse --git-path hooks/pre-push)"

cat > "$HOOK" <<'EOF'
#!/bin/bash
# Installed by scripts/install_git_hooks.sh - see that file for what this is.
if [ -n "$DILEMMA_SKIP_HF_CHECK" ]; then
    exit 0
fi
ROOT="$(git rev-parse --show-toplevel)"
if [ ! -f "$ROOT/scripts/hf_data.py" ]; then
    exit 0
fi
if ! python3 "$ROOT/scripts/hf_data.py" check; then
    echo
    echo "pre-push blocked: the HuggingFace data artifacts are out of sync." >&2
    echo "Push anyway with: git push --no-verify" >&2
    exit 1
fi
EOF
chmod +x "$HOOK"
echo "installed $HOOK"
