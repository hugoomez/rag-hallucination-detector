"""Unit tests for the --logging_steps / --max_grad_norm / --save_total_limit flags and their wiring.

These flags exist to support the ModernBERT-large fp16-stability experiment (tighter
loss logging + a gradient-clip fallback) and disk-space control across parallel runs.
The load-bearing property tested here is that leaving all of them at their defaults
reproduces exactly the historical TrainingArguments, so published base-model runs
stay bit-identical -- the new knobs are no-ops until used.
"""

from src.models.train_token_level import build_training_args, parse_args

# HF Trainer defaults, confirmed against transformers 4.57: an unset clip norm is 1.0,
# logging is step-based every 500 steps. The pre-change script forced logging_strategy
# to "epoch" (overriding the step cadence), so per-epoch logging is "today's" behavior.
HF_DEFAULT_MAX_GRAD_NORM = 1.0
HF_DEFAULT_LOGGING_STEPS = 500


def _args(tmp_path, *extra):
    # output_dir -> tmp so no models/ dir is created as a construction side effect.
    return parse_args(["--output_dir", str(tmp_path), *extra])


def test_new_flags_parse_to_noop_defaults(tmp_path):
    args = _args(tmp_path)
    assert args.logging_steps is None
    assert args.max_grad_norm == HF_DEFAULT_MAX_GRAD_NORM


def test_defaults_leave_training_args_unchanged(tmp_path):
    ta = build_training_args(_args(tmp_path))
    # The two new knobs must land on exactly the historical / HF-default configuration.
    assert ta.max_grad_norm == HF_DEFAULT_MAX_GRAD_NORM
    assert ta.logging_strategy == "epoch"
    assert ta.logging_steps == HF_DEFAULT_LOGGING_STEPS  # untouched: ignored under "epoch"
    # Sanity: the rest of the recipe still flows through from args as before.
    assert ta.learning_rate == 2e-5
    assert ta.per_device_train_batch_size == 4
    assert ta.num_train_epochs == 8.0
    assert ta.fp16 is True
    assert ta.gradient_checkpointing is True
    assert ta.metric_for_best_model == "eval_response_f1"


def test_logging_steps_switches_to_step_strategy(tmp_path):
    ta = build_training_args(_args(tmp_path, "--logging_steps", "10"))
    assert ta.logging_strategy == "steps"
    assert ta.logging_steps == 10
    # Only the logging cadence changes; clipping stays at the default no-op.
    assert ta.max_grad_norm == HF_DEFAULT_MAX_GRAD_NORM


def test_max_grad_norm_override(tmp_path):
    ta = build_training_args(_args(tmp_path, "--max_grad_norm", "0.5"))
    assert ta.max_grad_norm == 0.5
    # Only clipping changes; logging cadence stays at the historical per-epoch default.
    assert ta.logging_strategy == "epoch"


def test_save_total_limit_default_leaves_training_args_unchanged(tmp_path):
    # save_total_limit=1 has been hardcoded here since this script's first version;
    # the default must reproduce that exact value, not HF's own unlimited default.
    ta = build_training_args(_args(tmp_path))
    assert ta.save_total_limit == 1


def test_save_total_limit_override(tmp_path):
    ta = build_training_args(_args(tmp_path, "--save_total_limit", "2"))
    assert ta.save_total_limit == 2
    # Only the checkpoint cap changes; the rest of the recipe is untouched.
    assert ta.max_grad_norm == HF_DEFAULT_MAX_GRAD_NORM
    assert ta.logging_strategy == "epoch"
