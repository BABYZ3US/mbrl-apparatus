"""The studio boundary must stay torch-free — the seal (docs/remote_execution.md §1).

Imports the boundary server + every ``mbrl.studio.*`` module in a FRESH subprocess and
asserts torch / numpy / wandb never got imported. A single `from mbrl.utils import …`
(`mbrl.utils.__init__` pulls torch) would fail this — which is the whole point. The
existing `make seal-check` greps the TRAINING entrypoints for analysis deps (the
opposite direction) and never covers the boundary; this closes that gap.
"""
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Everything reachable from the one boundary; none may pull a heavy ML dep at import.
_SEAL_MODULES = [
    "studio_bridge_server",          # scripts/ — the boundary server itself
    "mbrl.studio.protocol",
    "mbrl.studio.run_index",
    "mbrl.studio.surface_index",
    "mbrl.studio.sweep",
    "mbrl.studio.spec_validator",
    "mbrl.studio.spec_to_config",
    "mbrl.studio.metric_db",
    "mbrl.studio.launch",
    "mbrl.studio.artifacts",
]
_FORBIDDEN = ("torch", "numpy", "wandb")


def test_studio_boundary_imports_are_torch_free():
    code = (
        "import sys, importlib\n"
        f"sys.path.insert(0, {str(_REPO / 'scripts')!r})\n"
        f"sys.path.insert(0, {str(_REPO / 'src')!r})\n"
        f"for m in {_SEAL_MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        f"bad = [m for m in {_FORBIDDEN!r} if m in sys.modules]\n"
        "assert not bad, 'seal broken — boundary imported: ' + ','.join(bad)\n"
        "print('SEAL OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"seal test failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "SEAL OK" in r.stdout
