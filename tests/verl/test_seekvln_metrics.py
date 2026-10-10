import pytest

from llava.verl.seekvln_metrics import aggregate_seekvln_decision_metrics


def test_seekvln_metrics_aggregate_modes_format_and_vln_metrics():
    decisions = [
        {
            "episode_id": "ep1",
            "decision_index": 0,
            "mode": "nav",
            "format_valid": True,
            "current": {"success": 0.0, "spl": 0.0, "ndtw": 0.2, "oracle_success": 0.0},
        },
        {
            "episode_id": "ep1",
            "decision_index": 1,
            "mode": "seek",
            "format_valid": False,
            "invalid_reason": "missing_key_evidence",
            "current": {"success": 1.0, "spl": 0.5, "ndtw": 0.7, "oracle_success": 1.0},
        },
        {
            "episode_id": "ep2",
            "decision_index": 0,
            "mode": "seek",
            "format_valid": True,
            "current": {"success": 0.0, "spl": 0.0, "ndtw": 0.3, "oracle_success": 1.0},
        },
    ]

    metrics = aggregate_seekvln_decision_metrics(decisions)

    assert metrics["seekvln/episode_count"] == 2.0
    assert metrics["seekvln/decision_count"] == 3.0
    assert metrics["seekvln/nav_rate"] == pytest.approx(1 / 3)
    assert metrics["seekvln/seek_rate"] == pytest.approx(2 / 3)
    assert metrics["seekvln/format_accuracy"] == pytest.approx(2 / 3)
    assert metrics["seekvln/invalid_trigger_rate"] == pytest.approx(1 / 3)
    assert metrics["seekvln/invalid_reason/missing_key_evidence"] == 1.0
    assert metrics["seekvln/sr"] == pytest.approx(0.5)
    assert metrics["seekvln/spl"] == pytest.approx(0.25)
    assert metrics["seekvln/ndtw"] == pytest.approx(0.5)
    assert metrics["seekvln/osr"] == pytest.approx(1.0)


def test_seekvln_metrics_empty_input_is_well_defined():
    metrics = aggregate_seekvln_decision_metrics([])
    assert metrics["seekvln/episode_count"] == 0.0
    assert metrics["seekvln/decision_count"] == 0.0
    assert metrics["seekvln/invalid_trigger_rate"] == 0.0
