"""Unit tests for the ACWS weighted token loss (weighted_token_ce).

Pins the three properties the ablation relies on, all on tiny CPU tensors (no model):
  1. implicit_true_weight == 1.0 reduces EXACTLY to the plain mean CE (arm b == today's loss);
  2. implicit_true_weight == 0.0 is EXACTLY loss-masking the flagged tokens;
  3. the per-token weight vector math (and -100-over-flag precedence) is correct for 0<lambda<1.
"""

import torch
import torch.nn as nn

from src.data.preprocess_token_level import IGNORE_LABEL
from src.models.train_token_level import NUM_LABELS, weighted_token_ce

torch.manual_seed(0)


def _sample():
    # (B=2, seq=5, classes=2) logits; -100 marks context/padding; mask flags some real tokens.
    logits = torch.randn(2, 5, NUM_LABELS)
    labels = torch.tensor([[0, 1, 1, IGNORE_LABEL, IGNORE_LABEL], [1, 0, 1, IGNORE_LABEL, IGNORE_LABEL]])
    mask = torch.tensor([[0, 1, 0, 0, 1], [0, 0, 1, 0, 0]])  # note: one flag sits at an IGNORE position
    return logits, labels, mask


def test_lambda_one_reduces_to_plain_mean_ce():
    logits, labels, mask = _sample()
    got = weighted_token_ce(logits, labels, mask, implicit_true_weight=1.0)
    ref = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)(logits.view(-1, NUM_LABELS), labels.view(-1))
    assert torch.allclose(got, ref, atol=1e-7)


def test_lambda_zero_equals_masking_flagged_tokens():
    logits, labels, mask = _sample()
    got = weighted_token_ce(logits, labels, mask, implicit_true_weight=0.0)
    # Masking = set flagged real tokens to IGNORE_LABEL and take the plain mean CE.
    masked = labels.clone()
    masked[mask.bool()] = IGNORE_LABEL
    ref = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)(logits.view(-1, NUM_LABELS), masked.view(-1))
    assert torch.allclose(got, ref, atol=1e-7)


def test_weight_vector_math_and_ignore_precedence():
    logits, labels, mask = _sample()
    lam = 0.25
    got = weighted_token_ce(logits, labels, mask, implicit_true_weight=lam)

    per_token = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="none")(
        logits.view(-1, NUM_LABELS), labels.view(-1)
    )
    flat_labels = labels.view(-1)
    flat_mask = mask.view(-1).bool()
    # -100 weight 0 (overrides flag), else lambda if flagged else 1.
    weights = torch.where(flat_mask, torch.full_like(per_token, lam), torch.ones_like(per_token))
    weights = torch.where(flat_labels == IGNORE_LABEL, torch.zeros_like(weights), weights)
    expected = (per_token * weights).sum() / weights.sum().clamp(min=1e-8)
    assert torch.allclose(got, expected, atol=1e-7)


def test_flag_at_ignore_position_contributes_nothing():
    # The sample deliberately flags an IGNORE position (row 0, col 4). Its weight must be 0,
    # so clearing that flag leaves the loss unchanged.
    logits, labels, mask = _sample()
    with_flag = weighted_token_ce(logits, labels, mask, implicit_true_weight=0.25)
    mask_cleared = mask.clone()
    mask_cleared[0, 4] = 0
    without_flag = weighted_token_ce(logits, labels, mask_cleared, implicit_true_weight=0.25)
    assert torch.allclose(with_flag, without_flag, atol=1e-7)


def test_lambda_between_zero_and_one_lies_between():
    logits, labels, mask = _sample()
    loss0 = weighted_token_ce(logits, labels, mask, 0.0)
    loss_half = weighted_token_ce(logits, labels, mask, 0.5)
    loss1 = weighted_token_ce(logits, labels, mask, 1.0)
    lo, hi = sorted([loss0.item(), loss1.item()])
    assert lo - 1e-6 <= loss_half.item() <= hi + 1e-6
