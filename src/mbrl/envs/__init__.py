from .tasks import make_task_env, task_split, FAMILIES, TaskWrapper

# ---- gym registration (idempotent): the in-house benchmark ----
# Registered at package import so `gym.make("TraceAtlas-v0")` works anywhere
# the apparatus is importable (train.py's make_env path included).
import gymnasium as _gym

if "TraceAtlas-v0" not in _gym.registry:
    _gym.register(
        id="TraceAtlas-v0",
        entry_point="mbrl.envs.trace_atlas:TraceAtlasReconstructionEnv",
    )
