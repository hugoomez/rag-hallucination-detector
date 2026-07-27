"""Unit tests for the --attn_implementation flag.

Unlike --fp16/--bf16, attn_implementation isn't threaded through build_training_args --
it goes straight from parsed args to AutoModelForTokenClassification.from_pretrained(...) in
main(), which isn't unit-testable without loading a real model (see build_training_args'
docstring for why that extraction exists for TrainingArguments but not model construction).
These tests pin what is testable in isolation: parse_args() resolves the right value for the
default and an explicit choice, and rejects anything outside {sdpa, eager} before it could
ever reach from_pretrained.
"""

import pytest

from src.models.train_token_level import parse_args


def test_default_attn_implementation_is_sdpa():
    args = parse_args([])
    assert args.attn_implementation == "sdpa"  # unchanged existing Kaggle T4 behavior


def test_explicit_eager_is_passed_through():
    args = parse_args(["--attn_implementation", "eager"])
    assert args.attn_implementation == "eager"


def test_invalid_attn_implementation_choice_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--attn_implementation", "flash_attention_2"])
