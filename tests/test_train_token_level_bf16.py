"""Unit tests for the --bf16 flag (mirrors --fp16's BooleanOptionalAction wiring).

build_training_args() is exercised directly on parsed argparse.Namespace objects, no
model/data loading required (see train_token_level.build_training_args docstring).

TrainingArguments.__post_init__ validates bf16 against actual hardware (Ampere+ GPU or
CPU+torch>=1.10) and raises on this CPU-only, non-Ampere dev/CI box. That hardware check is
transformers' concern, not ours -- these tests only pin our own arg-wiring logic (which flag
value reaches TrainingArguments), so is_torch_bf16_gpu_available is patched to simulate a
bf16-capable GPU, matching what a real Kaggle A100/L4 run would see.
"""

from unittest.mock import patch

from src.models.train_token_level import build_training_args, parse_args


def test_default_leaves_training_args_unchanged():
    args = parse_args([])
    ta = build_training_args(args)
    assert ta.bf16 is False
    assert ta.fp16 is True  # unchanged existing default (Kaggle T4 runs)


@patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=True)
def test_bf16_flag_sets_bf16_true(_mock_bf16_available):
    args = parse_args(["--bf16"])
    ta = build_training_args(args)
    assert ta.bf16 is True


@patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=True)
def test_bf16_takes_precedence_over_fp16(_mock_bf16_available):
    # --fp16 defaults to True, so passing --bf16 alone already conflicts unless bf16 wins.
    args = parse_args(["--bf16"])
    ta = build_training_args(args)
    assert ta.bf16 is True
    assert ta.fp16 is False

    # Explicit --fp16 alongside --bf16 must not override the precedence.
    args_explicit = parse_args(["--bf16", "--fp16"])
    ta_explicit = build_training_args(args_explicit)
    assert ta_explicit.bf16 is True
    assert ta_explicit.fp16 is False


def test_no_bf16_no_fp16_leaves_both_off():
    args = parse_args(["--no-fp16"])
    ta = build_training_args(args)
    assert ta.bf16 is False
    assert ta.fp16 is False
