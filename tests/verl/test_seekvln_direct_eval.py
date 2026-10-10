import torch

from scripts.eval.seekvln_auxthink_eval import SeekVLNDirectEvaluator, _generated_token_ids
from llava.verl.seekvln_inference import parse_seekvln_response


def test_generated_token_ids_accepts_full_sequence_output():
    prompt = torch.tensor([[11, 12, 13]])
    output = torch.tensor([[11, 12, 13, 21, 22]])

    assert _generated_token_ids(output, prompt) == [21, 22]


def test_generated_token_ids_accepts_generated_only_output():
    prompt = torch.tensor([[11, 12, 13]])
    output = torch.tensor([[21, 22]])

    assert _generated_token_ids(output, prompt) == [21, 22]


def test_direct_eval_executes_auxthink_action_for_malformed_seek():
    evaluator = object.__new__(SeekVLNDirectEvaluator)
    evaluator.invalid_action_policy = "auxthink_random"
    parsed = parse_seekvln_response("seek", "Turn left 45 degree.")

    actions, invalid_action_executed, fallback_text = evaluator._executable_actions(parsed)

    assert not parsed.format_valid
    assert parsed.invalid_reason == "missing_think_close_nav"
    assert parsed.actions == []
    assert actions == [2, 2, 2]
    assert invalid_action_executed
    assert fallback_text == "Turn left 45 degree."


def test_direct_eval_random_fallback_for_unparseable_malformed_seek():
    evaluator = object.__new__(SeekVLNDirectEvaluator)
    evaluator.invalid_action_policy = "auxthink_random"
    parsed = parse_seekvln_response("seek", "I am unsure.")

    actions, invalid_action_executed, fallback_text = evaluator._executable_actions(parsed)

    assert not parsed.format_valid
    assert parsed.invalid_reason == "missing_think_close_nav"
    assert parsed.actions == []
    assert len(actions) == 1
    assert actions[0] in {1, 2, 3}
    assert invalid_action_executed
    assert fallback_text == "I am unsure."
