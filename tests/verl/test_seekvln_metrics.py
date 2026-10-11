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


@pytest.mark.parametrize("distance", [float("inf"), float("nan"), None])
def test_undefined_distance_does_not_drop_failure_from_other_metrics(distance):
    records = [
        {"episode_id": "1", "done": True, "distance_to_goal": 4.9,
         "success": 1.0, "spl": 0.5, "ndtw": 0.8},
        {"episode_id": "2", "done": True, "distance_to_goal": distance,
         "success": 0.0, "spl": 0.0, "ndtw": 0.0},
    ]
    result = aggregate_seekvln_decision_metrics(records)
    assert result["seekvln/ne"] == 4.9
    assert result["seekvln/ne_episode_count"] == 1.0
    assert result["seekvln/ne_undefined_episode_count"] == 1.0
    assert result["seekvln/sr"] == 0.5
    assert result["seekvln/spl"] == 0.25
    assert result["seekvln/ndtw"] == 0.4


def test_unreachable_failed_episode_nan_spl_counts_as_zero():
    result = aggregate_seekvln_decision_metrics([
        {"episode_id": "1", "distance_to_goal": 2.0, "success": 1.0, "spl": 0.6},
        {"episode_id": "2", "distance_to_goal": float("inf"), "success": 0.0,
         "spl": float("nan")},
    ])
    assert result["seekvln/ne"] == 2.0
    assert result["seekvln/spl"] == 0.3
    assert result["seekvln/spl_episode_count"] == 2.0
    assert result["seekvln/spl_zero_filled_episode_count"] == 1.0


def test_nonfinite_metric_is_reported_not_silently_zero_filled():
    result = aggregate_seekvln_decision_metrics([
        {"episode_id": "1", "distance_to_goal": 2.0, "success": 1.0, "spl": float("nan")},
    ])
    assert "seekvln/spl" not in result
    assert result["seekvln/spl_undefined_episode_count"] == 1.0


def test_partial_episode_is_not_a_terminal_metric():
    result = aggregate_seekvln_decision_metrics([
        {"episode_id": "1", "done": True, "success": 1.0},
        {"episode_id": "2", "done": False, "success": 0.0},
    ])
    assert result["seekvln/episode_count"] == 2.0
    assert result["seekvln/incomplete_episode_count"] == 1.0
    assert result["seekvln/sr"] == 1.0
    assert result["seekvln/sr_episode_count"] == 1.0


def test_disjoint_shards_and_repeated_decisions_use_one_final_per_episode():
    shard1 = [{"episode_id": "1", "decision_index": 0, "success": 0.0},
              {"episode_id": "1", "decision_index": 1, "success": 1.0}]
    shard2 = [{"episode_id": "2", "decision_index": 0, "success": 0.0}]
    result = aggregate_seekvln_decision_metrics(shard2 + shard1)
    assert result["seekvln/episode_count"] == 2.0
    assert result["seekvln/sr"] == 0.5
