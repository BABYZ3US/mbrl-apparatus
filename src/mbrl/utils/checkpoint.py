"""CheckpointManager — atomic, resumable, W&B-artifact-aware (PLAN.md section 4).

Captures everything needed for bitwise-identical resume: module/optimizer state
(via the trainer's state_dict protocol), lambda-schedule step, RNG states
(torch/numpy/python), env step count, and a config hash that is verified on
resume. Designed around session death / spot preemption: save every `every` updates and on
SIGTERM; `resume="auto"` restores the newest checkpoint for the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch


def config_hash(cfg_dict: dict) -> str:
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


class CheckpointManager:
    def __init__(self, dirpath: str | Path, cfg_dict: dict, every: int = 2000,
                 keep_last: int = 3, milestone_every: int | None = 100_000,
                 wandb_run=None):
        self.hash = config_hash(cfg_dict)
        # Scope by config hash: a config change starts a FRESH checkpoint lineage
        # instead of colliding with (or crashing on) checkpoints from an older
        # config. Old lineages are preserved in sibling dirs.
        self.dir = Path(dirpath) / self.hash
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every, self.keep_last, self.milestone_every = every, keep_last, milestone_every
        self.wandb_run = wandb_run
        self._best = -float("inf")
        self._install_sigterm = False  # set True in scripts; not in notebooks/tests

    # ---------------- save ----------------
    def maybe_save(self, trainer, env_steps: int, eval_return: float | None = None):
        if trainer.step % self.every == 0:
            self.save(trainer, env_steps, tag=f"step{trainer.step}")
        if eval_return is not None and eval_return > self._best:
            self._best = eval_return
            self.save(trainer, env_steps, tag="best")
        if self.milestone_every and env_steps and env_steps % self.milestone_every == 0:
            self.save(trainer, env_steps, tag=f"milestone{env_steps}", permanent=True)

    def save(self, trainer, env_steps: int, tag: str, permanent: bool = False):
        payload = {
            "trainer": trainer.state_dict(),
            "env_steps": env_steps,
            "cfg_hash": self.hash,
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        path = self.dir / f"ckpt_{tag}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic
        if not permanent and tag not in ("best",):
            self._gc()
        if self.wandb_run is not None:
            import wandb
            art = wandb.Artifact(f"model-{self.wandb_run.id}", type="checkpoint",
                                 metadata={"tag": tag, "env_steps": env_steps,
                                           "cfg_hash": self.hash})
            art.add_file(str(path))
            self.wandb_run.log_artifact(art)
        return path

    def _gc(self):
        steps = sorted(self.dir.glob("ckpt_step*.pt"),
                       key=lambda p: int(p.stem.split("step")[1]))
        for p in steps[:-self.keep_last]:
            p.unlink()

    # ---------------- restore ----------------
    def resume(self, trainer, mode: str = "auto") -> int:
        """Restore trainer + RNG. Returns env_steps (0 if nothing to resume).

        mode="auto": only sees this config's lineage (hash-scoped dir), so a
        config change silently starts fresh — never crashes.
        mode=<path>: explicit checkpoint; hash mismatch is a hard error, since
        the user asked for that specific file."""
        path = self._latest() if mode == "auto" else Path(mode)
        if path is None or not path.exists():
            if mode == "auto":
                others = [d.name for d in self.dir.parent.iterdir()
                          if d.is_dir() and d.name != self.hash] \
                    if self.dir.parent.exists() else []
                if others:
                    print(f"[checkpoint] no lineage for cfg {self.hash}; starting "
                          f"fresh (other-config lineages present: {others})")
            return 0
        payload = torch.load(path, weights_only=False)
        if payload["cfg_hash"] != self.hash:
            raise RuntimeError(
                f"Refusing to resume explicit checkpoint {path}: cfg_hash "
                f"{payload['cfg_hash']} != current {self.hash}. Pass the matching "
                f"config or resume=auto to start a fresh lineage.")
        trainer.load_state_dict(payload["trainer"])
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        return payload["env_steps"]

    def _latest(self) -> Path | None:
        cands = list(self.dir.glob("ckpt_step*.pt"))
        return max(cands, key=lambda p: int(p.stem.split("step")[1])) if cands else None

    def fetch_from_wandb(self, run_path: str, tag: str = "latest") -> Path:
        """Mode-B: pull the newest checkpoint artifact (e.g. on worker relaunch)."""
        import wandb
        api = wandb.Api()
        art = api.artifact(f"{run_path}:{tag}", type="checkpoint")
        return Path(art.download(root=str(self.dir)))

    # ---------------- SIGTERM guard ----------------
    def install_signal_handler(self, trainer, get_env_steps):
        def _handler(signum, frame):
            self.save(trainer, get_env_steps(), tag=f"step{trainer.step}")
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, _handler)
