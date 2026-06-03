from src.training.ppo_trainer import latest_metrics as latest_single_metrics
from src.training.team_ppo_trainer import latest_metrics as latest_team_metrics


def test_latest_metrics_accepts_bom_prefixed_jsonl_lines(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"update": 1}\n\ufeff{"update": 2}\n', encoding="utf-8")

    assert latest_single_metrics(metrics_path) == [{"update": 1}, {"update": 2}]
    assert latest_team_metrics(metrics_path) == [{"update": 1}, {"update": 2}]
