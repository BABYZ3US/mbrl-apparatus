"""Atomic, namespaced result-envelope writer (resultio).

Phase-3 fix for the audit finding that Simulations / verification jobs wrote
their results to *fixed* filenames (e.g. ``result.json``). When several agents
run in parallel they all targeted the same path, so the last writer clobbered
the others' output and partially-written files were observable to readers.

This module fixes both failure modes:

* **Namespacing** — every result gets a time-sortable, collision-resistant id
  (:func:`new_id`), and its filename (:func:`result_filename`) is derived from
  ``task`` *and* ``id``. Two results for the same task can never share a path,
  so parallel agents never collide.

* **Atomicity** — :func:`write` serializes to a unique temp file in the *same*
  directory, ``flush`` + ``fsync`` it, then ``os.replace`` it onto the final
  name. ``os.replace`` is an atomic rename within a directory on POSIX/NTFS, so
  a reader either sees the old file or the fully-written new one — never a
  truncated/partial file, and never a half-written clobber.

Built on top of the generated typed envelope at
``mbrl.studio._spine_result`` (source of truth: ``spine/`` codegen). This
module re-exports :class:`ResultEnvelope` and :data:`STATUS` so callers have a
single import site.

Seal-safe: standard library only (``dataclasses, json, os, tempfile, time,
uuid, pathlib, typing``) — no torch / numpy / pydantic / yaml.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from ..studio._spine_result import STATUS, ResultEnvelope

__all__ = [
    "ResultEnvelope",
    "STATUS",
    "new_id",
    "now_ts",
    "make",
    "result_filename",
    "write",
    "read",
]

# Fields of ResultEnvelope that callers may pass through ``make`` (everything
# except the auto-filled id/ts and the required task/status).
_PASSTHROUGH = ("seed", "params", "value", "evidence", "blockers", "method", "checked")


def new_id(prefix: str = "") -> str:
    """Return a time-sortable, collision-resistant id.

    Layout: ``<prefix><ms-since-epoch, 13 digits>-<8 hex of a uuid4>``. The
    millisecond timestamp makes ids lexicographically sortable by creation
    time; the uuid4 suffix makes them collision-resistant even when many are
    minted within the same millisecond (or across parallel processes). This is
    never a fixed value, which is the whole point of the clobber fix.
    """
    return f"{prefix}{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:8]}"


def now_ts() -> str:
    """Return the current time as an ISO-8601 UTC timestamp (second precision)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make(task: str, status: str, **fields: Any) -> ResultEnvelope:
    """Build a :class:`ResultEnvelope`, validating ``status`` and auto-filling ids.

    ``status`` must be a member of :data:`STATUS` or a :class:`ValueError` is
    raised (listing the allowed values). ``id`` and ``ts`` are auto-filled via
    :func:`new_id` / :func:`now_ts` when not supplied. The optional envelope
    fields (seed, params, value, evidence, blockers, method, checked) are
    passed through; any unknown keyword is rejected with a :class:`TypeError`.
    """
    if status not in STATUS:
        raise ValueError(
            f"invalid status {status!r}; must be one of {tuple(STATUS)}"
        )

    unknown = set(fields) - set(_PASSTHROUGH) - {"id", "ts"}
    if unknown:
        raise TypeError(
            f"make() got unexpected field(s) {sorted(unknown)}; "
            f"allowed: {sorted(_PASSTHROUGH)}"
        )

    env_id = fields.pop("id", None) or new_id()
    ts = fields.pop("ts", None) or now_ts()

    kwargs = {k: v for k, v in fields.items() if k in _PASSTHROUGH}
    return ResultEnvelope(id=env_id, task=task, status=status, ts=ts, **kwargs)


def _sanitize(text: str) -> str:
    """Reduce arbitrary text to a filesystem-safe token (alnum / ``-`` / ``_``)."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in text)
    # Collapse runs of separators and trim, so names stay tidy.
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-_") or "result"


def result_filename(env: Union[ResultEnvelope, dict]) -> str:
    """Return a filesystem-safe, namespaced ``.json`` name for ``env``.

    The name combines the sanitized task with the (already collision-resistant)
    id: ``<safe_task>__<id>.json``. Two results with the same task but distinct
    ids therefore get distinct filenames — no clobbering.
    """
    task = env["task"] if isinstance(env, dict) else env.task
    env_id = env["id"] if isinstance(env, dict) else env.id
    return f"{_sanitize(str(task))}__{_sanitize(str(env_id))}.json"


def _as_dict(env: Union[ResultEnvelope, dict]) -> dict:
    if isinstance(env, dict):
        return env
    return env.to_dict()


def write(
    env: Union[ResultEnvelope, dict],
    out_dir: Union[str, os.PathLike],
    *,
    filename: Optional[str] = None,
) -> Path:
    """Atomically write ``env`` (envelope or plain dict) as JSON into ``out_dir``.

    The directory is created if needed. JSON is written to a unique temp file
    in ``out_dir``, flushed and ``fsync``-ed, then ``os.replace``-d onto the
    final name (an atomic rename within the directory). No partial or clobbered
    file is ever observable. Returns the final :class:`Path`.
    """
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    payload = _as_dict(env)
    name = filename if filename is not None else result_filename(payload)
    final = out_dir / name

    # NamedTemporaryFile(dir=out_dir) guarantees the temp file is on the same
    # filesystem as ``final`` so os.replace is a true atomic rename.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_dir,
        prefix=".tmp-",
        suffix=".json",
        delete=False,
    )
    try:
        with tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, final)
    except BaseException:
        # Best-effort cleanup so a failed write leaves no stray temp file.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return final


def read(path: Union[str, os.PathLike]) -> ResultEnvelope:
    """Load a JSON result file and reconstruct a :class:`ResultEnvelope`."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return ResultEnvelope(**data)
