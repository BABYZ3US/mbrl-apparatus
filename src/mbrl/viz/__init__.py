"""Viz package — LAZY submodule access (PEP 562).

curves/reward_surface/latent_space need matplotlib (the `analysis` extra,
deliberately absent from CI and the sealed image); surface_export is
stdlib+numpy and ships inside the seal. Eager imports here made importing
ANY viz module require matplotlib — caught by CI run 1, invisible locally
where the extra is installed. Submodules now import on first attribute
access; `from mbrl.viz import curves` still works unchanged.
"""
import importlib

_SUBMODULES = ("curves", "reward_surface", "latent_space", "surface_export")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES))
