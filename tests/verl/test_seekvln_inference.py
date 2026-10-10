from llava.verl.seekvln_inference import (
    SEEK_ENVIRONMENT_TURN,
    build_seekvln_stage1_prompt,
    parse_seekvln_action,
    parse_seekvln_response,
)


def test_seekvln_stage1_prompt_matches_training_image_contract():
    prompt = build_seekvln_stage1_prompt("Go to the sofa", 3)
    assert prompt.count("<image>") == 3
    assert "historical observations: <image><image> and current observation:<image>." in prompt
    assert "begin your response with exactly one mode token: <nav> or <seek>" in prompt


def test_parse_direct_nav_response():
    parsed = parse_seekvln_response("nav", "The next action is move forward 75 cm.")
    assert parsed.mode == "nav"
    assert parsed.format_valid
    assert parsed.invalid_reason is None
    assert parsed.actions == [1, 1, 1]
    assert parsed.text.startswith("<nav>")


def test_parse_action_matches_auxthink_floor_and_substring_rules():
    assert parse_seekvln_action("I think I should forward 75 cm.") == [1, 1, 1]
    assert parse_seekvln_action("The next action is turn left 25 degrees.") == [2]
    assert parse_seekvln_action("The next action is turn right 45 degrees, then move forward 25 cm.") == [3, 3, 3]


def test_parse_action_executes_only_the_first_macro_action_span():
    # A response is one environment decision.  Distance/angle expansion within
    # that first span is allowed, but a later comma-separated macro is ignored.
    assert parse_seekvln_action(
        "turn right 45 degrees, then move forward 75 cm, then stop"
    ) == [3, 3, 3]


def test_parse_seek_response_requires_structure_and_action():
    tail = (
        "Completed tasks: Exit the bedroom. Next task: Find the dining table. "
        "Key Evidence: I see the table.</think><nav>The next action is turn left 30 degrees."
    )
    parsed = parse_seekvln_response("seek", tail)
    assert SEEK_ENVIRONMENT_TURN in parsed.text
    assert parsed.format_valid
    assert parsed.invalid_reason is None
    assert parsed.actions == [2, 2]
    assert parsed.final_action_text == "The next action is turn left 30 degrees."


def test_parse_seek_commas_inside_think_do_not_truncate_final_action():
    tail = (
        "Completed tasks: Exit the bedroom, pass the sofa. Next task: Find the dining table. "
        "Key Evidence: I see chairs, table legs, and a dining area.</think><nav>"
        "The next action is move forward 75 cm."
    )
    parsed = parse_seekvln_response("seek", tail)
    assert parsed.format_valid
    assert parsed.actions == [1, 1, 1]
    assert "chairs, table legs" in parsed.text


def test_parse_seek_response_uses_only_first_macro_after_nav():
    tail = (
        "Completed tasks: Exit. Next task: Find the table. Key Evidence: table visible."
        "</think><nav>turn left 30 degrees, then move forward 75 cm"
    )
    parsed = parse_seekvln_response("seek", tail)
    assert parsed.format_valid
    assert parsed.actions == [2, 2]


def test_parse_nav_invalid_reason_for_unparseable_action():
    parsed = parse_seekvln_response("nav", "I am unsure.")
    assert not parsed.format_valid
    assert parsed.invalid_reason == "unparseable_action"
    assert parsed.action_fallback
    assert len(parsed.actions) == 1
    assert parsed.actions[0] in {1, 2, 3}


def test_parse_seek_invalid_reasons_are_specific():
    missing_nav = parse_seekvln_response(
        "seek",
        "Completed tasks: Exit. Next task: Stop. Key Evidence: I see the target.",
    )
    assert not missing_nav.format_valid
    assert missing_nav.invalid_reason == "missing_think_close_nav"

    missing_evidence = parse_seekvln_response(
        "seek",
        "Completed tasks: Exit. Next task: Stop.</think><nav>The next action is stop.",
    )
    assert not missing_evidence.format_valid
    assert missing_evidence.invalid_reason == "missing_key_evidence"
