"""Smoke test: the diagnostics dashboard builds from whatever training state
exists right now (a live grid may be mid-write; sections degrade to notes —
the page itself must always generate)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECTION_IDS = ["sec-lambda-explorer", "sec-training-pulse", "sec-transversality",
               "sec-latent-density", "sec-reward-tensor"]


def test_dashboard_builds():
    spec = importlib.util.spec_from_file_location(
        "make_dashboard", ROOT / "scripts" / "make_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main([])  # default args — must tolerate any training state

    page = ROOT / "results" / "dashboard.html"
    assert page.exists(), "dashboard.html was not written"
    assert Path(out) == page
    html = page.read_text()
    assert len(html.encode()) > 50_000, "dashboard suspiciously small"
    assert "plotly" in html.lower()
    for sid in SECTION_IDS:
        assert sid in html, f"missing section id {sid}"
