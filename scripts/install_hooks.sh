#!/usr/bin/env bash
# install_hooks.sh (mbrl-curvature) — wire the version-controlled hooks/ dir into git.
#
# Points git at <repo>/hooks via core.hooksPath (so the hooks live in the repo and
# are reviewed like any other code), and marks them executable. Idempotent: safe to
# re-run. Also available as `make hooks-install`.
#
#   ./scripts/install_hooks.sh            # install (set core.hooksPath=hooks)
#   ./scripts/install_hooks.sh --uninstall  # revert to git's default .git/hooks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_DIR"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "[install_hooks] ERROR: $REPO_DIR is not a git repo." >&2
    exit 1
fi

if [ "${1:-}" = "--uninstall" ]; then
    git config --unset core.hooksPath 2>/dev/null || true
    echo "[install_hooks] Uninstalled: core.hooksPath unset (back to .git/hooks)."
    exit 0
fi

HOOKS_DIR="$REPO_DIR/hooks"
if [ ! -d "$HOOKS_DIR" ]; then
    echo "[install_hooks] ERROR: $HOOKS_DIR not found." >&2
    exit 1
fi

# Make every hook executable, then register the dir.
chmod +x "$HOOKS_DIR"/* 2>/dev/null || true
git config core.hooksPath hooks

echo "[install_hooks] core.hooksPath -> hooks"
echo "[install_hooks] active hooks:"
for h in "$HOOKS_DIR"/*; do
    [ -f "$h" ] || continue
    case "$(basename "$h")" in
        *.sample) continue ;;
    esac
    printf '  - %s%s\n' "$(basename "$h")" "$([ -x "$h" ] && echo '' || echo '  (NOT executable!)')"
done
echo "[install_hooks] Done. pre-push will run ruff + fast pytest before every push."
