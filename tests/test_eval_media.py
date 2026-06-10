"""W11: eval frames -> local GIF + manifest entry (the Studio-browsable leg)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.eval import save_eval_media
from mbrl.studio.artifacts import list_artifacts


def _frames(n=8, w=32, h=24):
    return [np.full((h, w, 3), i * 30, dtype=np.uint8) for i in range(n)]


def test_save_writes_gif_and_manifest_entry(tmp_path):
    out = save_eval_media(_frames(), tmp_path, "run_x", env_steps=5000)
    assert out.exists() and out.stat().st_size > 0
    assert out.name == "eval_s5000.gif"
    entries = list_artifacts(tmp_path, "run_x")
    vid = [e for e in entries if e["name"] == "eval_video"]
    assert len(vid) == 1
    assert vid[0]["type"] == "video" and vid[0]["step"] == 5000
    assert vid[0]["path"].endswith("eval_s5000.gif")


def test_upsert_keeps_the_latest_video_only(tmp_path):
    save_eval_media(_frames(), tmp_path, "run_x", env_steps=1000)
    save_eval_media(_frames(), tmp_path, "run_x", env_steps=2000)
    vid = [e for e in list_artifacts(tmp_path, "run_x") if e["name"] == "eval_video"]
    assert len(vid) == 1 and vid[0]["step"] == 2000          # the checkpoint convention
    # both FILES remain on disk (history browsable); the manifest points at latest
    media = list((tmp_path / "runs" / "run_x" / "media").glob("*.gif"))
    assert len(media) == 2


def test_empty_frames_refuse(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        save_eval_media([], tmp_path, "run_x", env_steps=0)
