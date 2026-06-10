"""W11 eval media: episode frames -> a LOCAL artifact the Studio can browse.

The W&B video path (train.py) is cloud-side; the bridge boundary deliberately
cannot reach it (the seal). This writes the SAME frames as a local GIF under
the run's media/ dir and upserts an artifact-manifest entry — so the Artifacts
panel lists it and "Open" plays it. GIF because imageio encodes it with no
extra system deps (no ffmpeg on this machine); if imageio-ffmpeg lands later,
switching to mp4 is a format argument.
"""
from __future__ import annotations

from pathlib import Path

from ..studio.artifacts import record_artifact


def save_eval_media(frames, results_root, run_name: str, env_steps: int,
                    fps: int = 30) -> Path:
    """Write ``frames`` (HxWx3 uint8 arrays) as media/eval_s<steps>.gif and
    record the manifest entry (upsert-by-name: 'eval_video' stays the LATEST
    one — the checkpoint convention). Raises ImportError without imageio; the
    caller's existing warn-once shield handles it."""
    import imageio.v3 as iio

    if not frames:
        raise ValueError("no frames to save")
    media_dir = Path(results_root) / "runs" / run_name / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    out = media_dir / f"eval_s{int(env_steps)}.gif"
    iio.imwrite(out, list(frames), duration=1000.0 / max(1, fps), loop=0)
    record_artifact(results_root, run_name, {
        "name": "eval_video", "type": "video", "step": int(env_steps),
        "path": str(out)})
    return out
