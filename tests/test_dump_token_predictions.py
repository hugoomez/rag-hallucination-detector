"""Unit tests for scripts/dump_token_predictions.py's tokenizer-loading fallback and CLI
validation. No model download, no data files -- AutoTokenizer/AutoConfig are monkeypatched.

Covers the TokenizersBackend workaround (a checkpoint's own tokenizer_config.json can be
unreadable by a pinned transformers version -- reproduced directly against the local
seed123 checkpoint before this fix existed) and the --seed/--metrics_out wiring needed for
scripts/aggregate_seeds.py compatibility.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import dump_token_predictions as dtp  # noqa: E402

TOKENIZER_CLASS_ERROR = "Tokenizer class TokenizersBackend does not exist or is not currently imported."


class _FakeConfig:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type


class TestLoadTokenizer:
    def test_explicit_tokenizer_id_wins_and_skips_model_path_entirely(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            dtp.AutoTokenizer, "from_pretrained", lambda repo: calls.append(repo) or f"tok:{repo}"
        )
        result = dtp.load_tokenizer("some/local/checkpoint", tokenizer_id="explicit/override")
        assert result == "tok:explicit/override"
        assert calls == ["explicit/override"]  # model path never attempted

    def test_direct_load_success_never_consults_fallback_map(self, monkeypatch):
        monkeypatch.setattr(dtp.AutoTokenizer, "from_pretrained", lambda repo: f"tok:{repo}")
        monkeypatch.setattr(
            dtp.AutoConfig,
            "from_pretrained",
            lambda repo: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        result = dtp.load_tokenizer("hugoomezz/modernbert-ragtruth-token-level-binary", tokenizer_id=None)
        assert result == "tok:hugoomezz/modernbert-ragtruth-token-level-binary"

    def test_tokenizer_class_error_falls_back_via_model_family_map(self, monkeypatch):
        calls = []

        def fake_from_pretrained(repo):
            calls.append(repo)
            if repo == "models/large_seed123/checkpoint-6792":
                raise ValueError(TOKENIZER_CLASS_ERROR)
            return f"tok:{repo}"

        monkeypatch.setattr(dtp.AutoTokenizer, "from_pretrained", fake_from_pretrained)
        monkeypatch.setattr(dtp.AutoConfig, "from_pretrained", lambda repo: _FakeConfig("modernbert"))

        result = dtp.load_tokenizer("models/large_seed123/checkpoint-6792", tokenizer_id=None)

        assert result == f"tok:{dtp.MODEL_FAMILY_TOKENIZER_FALLBACK['modernbert']}"
        assert calls == ["models/large_seed123/checkpoint-6792", dtp.MODEL_FAMILY_TOKENIZER_FALLBACK["modernbert"]]

    def test_unmapped_model_type_reraises(self, monkeypatch):
        monkeypatch.setattr(
            dtp.AutoTokenizer,
            "from_pretrained",
            lambda repo: (_ for _ in ()).throw(ValueError(TOKENIZER_CLASS_ERROR)),
        )
        monkeypatch.setattr(dtp.AutoConfig, "from_pretrained", lambda repo: _FakeConfig("some_unmapped_family"))

        with pytest.raises(ValueError, match="TokenizersBackend"):
            dtp.load_tokenizer("some/checkpoint", tokenizer_id=None)

    def test_unrelated_value_error_reraises_without_consulting_config(self, monkeypatch):
        monkeypatch.setattr(
            dtp.AutoTokenizer,
            "from_pretrained",
            lambda repo: (_ for _ in ()).throw(ValueError("totally different problem")),
        )
        monkeypatch.setattr(
            dtp.AutoConfig,
            "from_pretrained",
            lambda repo: (_ for _ in ()).throw(AssertionError("should not be called")),
        )

        with pytest.raises(ValueError, match="totally different problem"):
            dtp.load_tokenizer("some/checkpoint", tokenizer_id=None)


class TestParseArgsSeedMetricsOutValidation:
    def test_metrics_out_without_seed_errors(self):
        with pytest.raises(SystemExit):
            dtp.parse_args(["--metrics_out", "results/foo_metrics.json"])

    def test_metrics_out_with_seed_is_accepted(self):
        args = dtp.parse_args(["--metrics_out", "results/foo_metrics.json", "--seed", "123"])
        assert args.metrics_out == "results/foo_metrics.json"
        assert args.seed == 123

    def test_neither_flag_is_fine(self):
        args = dtp.parse_args([])
        assert args.metrics_out is None
        assert args.seed is None
