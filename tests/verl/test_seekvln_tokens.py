import pytest

from llava.verl.seekvln_tokens import (
    SEEKVLN_CONTROL_TOKENS,
    SEEKVLN_MODE_TOKENS,
    SEEKVLN_STRUCT_TOKENS,
    ensure_seekvln_tokenizer,
    normalized_control_tokens,
)


class _Tokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab = {"<unk>": self.unk_token_id}
        self.additional_special_tokens = []
        self.padding_side = "left"

    def add_special_tokens(self, special_tokens, replace_additional_special_tokens=True):
        added = 0
        if replace_additional_special_tokens:
            self.additional_special_tokens = list(special_tokens["additional_special_tokens"])
        else:
            for token in special_tokens["additional_special_tokens"]:
                if token not in self.additional_special_tokens:
                    self.additional_special_tokens.append(token)
        for token in special_tokens["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def add_tokens(self, tokens, special_tokens=False):
        del special_tokens
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, token, add_special_tokens=False):
        del add_special_tokens
        return [self.convert_tokens_to_ids(token)]

    def get_vocab(self):
        return dict(self.vocab)


def test_seekvln_tokenizer_contract_uses_methodology33_tokens():
    tokenizer = _Tokenizer()
    tokenizer.add_tokens(["<image>", "<vila/video>"])
    token_ids = ensure_seekvln_tokenizer(tokenizer)
    assert tuple(token_ids.control) == SEEKVLN_CONTROL_TOKENS
    assert token_ids.mode == [token_ids.control[token] for token in SEEKVLN_MODE_TOKENS]
    assert token_ids.response_special == [token_ids.control[token] for token in SEEKVLN_STRUCT_TOKENS]
    assert token_ids.mode == [token_ids.control["<nav>"], token_ids.control["<seek>"]]
    assert token_ids.response_special == [
        token_ids.control["</seek>"],
        token_ids.control["<think>"],
        token_ids.control["</think>"],
        token_ids.control["<nav>"],
    ]
    assert tokenizer.padding_side == "right"
    assert tokenizer.convert_tokens_to_ids("<image>") != tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_ids("<vila/video>") != tokenizer.unk_token_id


def test_seekvln_tokenizer_rejects_legacy_view_tokens():
    with pytest.raises(ValueError, match="legacy tokens"):
        normalized_control_tokens(["<nav>", "<seek>", "</seek>", "<think>", "</think>", "<view_0>"])


def test_seekvln_tokenizer_preserves_auxthink_special_tokens():
    tokenizer = _Tokenizer()
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<image>", "<vila/video>"]}
    )
    ensure_seekvln_tokenizer(tokenizer)
    assert tokenizer.additional_special_tokens == [
        "<|im_start|>",
        "<|im_end|>",
        "<image>",
        "<vila/video>",
        "<nav>",
        "<seek>",
        "</seek>",
        "<think>",
        "</think>",
    ]
