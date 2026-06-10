

def test_run_config_round_trips_through_index_and_server(tmp_path):
    """W8: config.json (MetricsLogger dump) -> RunIndex.get_config -> the
    pull.artifacts reply. Old runs without the dump answer {} honestly."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mbrl.studio.run_index import RunIndex
    from mbrl.utils.metrics_logger import MetricsLogger

    cfg = {"model": {"latent_dim": 4, "encoder": "vae"}, "seed": 0}
    MetricsLogger(tmp_path, "run_x", meta={"group": "g"}, config=cfg)
    idx = RunIndex(tmp_path)
    assert idx.get_config("run_x") == cfg
    assert idx.get_config("never_ran") == {}
    # torn file -> {}
    (tmp_path / "runs" / "run_t").mkdir(parents=True)
    (tmp_path / "runs" / "run_t" / "config.json").write_text("{not json")
    assert idx.get_config("run_t") == {}
