"""Model directory and configuration management.

A model directory contains everything needed to load, train, and generate:

    my_model/
    ├── config.json          # all hyperparameters
    ├── spec.pt              # frozen amplifier spec
    ├── tokenizer.model      # sentencepiece tokenizer (optional)
    ├── checkpoint.pt        # latest trainable weights + optimizer
    ├── checkpoint_NNNN.pt   # numbered checkpoints
    └── metrics.jsonl        # training log

Usage:
    # Create from scratch
    cfg = ModelConfig.create(model_dir="my_model", data="/path/to/data")
    cfg.save()

    # Load existing
    cfg = ModelConfig.load("my_model")

    # Override with CLI args
    cfg.update_from_args(args)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Defaults for everything. One source of truth.
DEFAULTS = {
    "model": {
        "vocab_size": 1024,
        "core_dim": 48,
        "branch_lags": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
        "branch_temporal_mode": "current",
        "branch_temporal_lag_scale": 1.0,
        "residual_token_gate_mode": "none",
        "branch_router_mode": "none",
        "base_bigram_delta": "none",
        "trigram_sidecar": "none",
        "trigram_log_scale_init": 0.0,
        "residual_readout_delta_rank": 0,
        "residual_readout_delta_init_std": 0.02,
        "num_blocks": 9,
        "core_layers": 5,
        "core_type": "mingru",
        "core_expansion": 2.0,
        "residual_core": True,
        "residual_core_init": -2.0,
        "readout_rank": None,
        "smoothing": 0.25,
        "embedding_init": "spectral",
        "spectral_neighbors": 64,
        "lag_identity_base": 0.15,
        "fixed_dtype": "bfloat16",
    },
    "training": {
        "seq_len": 512,
        "batch_size": 256,
        "carry_chunks": 16,
        "bptt_chunks": 2,
        "num_steps": 7000,
        "learning_rate": 3e-3,
        "lr_schedule": "cosine",
        "min_lr": 3e-4,
        "warmup_steps": 100,
        "lr_hold_steps": 1500,
        "weight_decay": 1e-3,
        "hard_loss_gamma": 0.5,
        "hard_loss_cap": 5.0,
        "grad_clip": 1.0,
        "dropout": 0.0,
        "amplifier_dtype": "auto",
        "scan_backend": "auto",
        "gradient_checkpointing": False,
        "log_state_every": 200,
        "full_val_final": False,
    },
    "data": {
        "source": None,
        "storage_dtype": "uint16",
        "train_frac": 0.98,
        "max_tokens": None,
    },
    "spec": {
        "strategy": "auto",
        "workers": -1,
        "max_tokens": None,
    },
}

# Maps CLI arg names → (config section, config key)
# Only need entries where names differ or aren't obvious.
_ARG_MAP: dict[str, tuple[str, str]] = {
    "data": ("data", "source"),
    "vocab_size": ("model", "vocab_size"),
    "core_dim": ("model", "core_dim"),
    "branch_lags": ("model", "branch_lags"),
    "branch_temporal_mode": ("model", "branch_temporal_mode"),
    "branch_temporal_lag_scale": ("model", "branch_temporal_lag_scale"),
    "residual_token_gate_mode": ("model", "residual_token_gate_mode"),
    "branch_router_mode": ("model", "branch_router_mode"),
    "base_bigram_delta": ("model", "base_bigram_delta"),
    "trigram_sidecar": ("model", "trigram_sidecar"),
    "trigram_log_scale_init": ("model", "trigram_log_scale_init"),
    "residual_readout_delta_rank": ("model", "residual_readout_delta_rank"),
    "residual_readout_delta_init_std": ("model", "residual_readout_delta_init_std"),
    "num_blocks": ("model", "num_blocks"),
    "core_layers": ("model", "core_layers"),
    "core_type": ("model", "core_type"),
    "core_expansion": ("model", "core_expansion"),
    "residual_core": ("model", "residual_core"),
    "residual_core_init": ("model", "residual_core_init"),
    "readout_rank": ("model", "readout_rank"),
    "smoothing": ("model", "smoothing"),
    "embedding_init": ("model", "embedding_init"),
    "spectral_neighbors": ("model", "spectral_neighbors"),
    "lag_identity_base": ("model", "lag_identity_base"),
    "fixed_dtype": ("model", "fixed_dtype"),
    "storage_dtype": ("data", "storage_dtype"),
    "train_frac": ("data", "train_frac"),
    "data_max_tokens": ("data", "max_tokens"),
    "spec_strategy": ("spec", "strategy"),
    "spec_workers": ("spec", "workers"),
    "spec_max_tokens": ("spec", "max_tokens"),
    "max_tokens": ("spec", "max_tokens"),
    "seq_len": ("training", "seq_len"),
    "batch_size": ("training", "batch_size"),
    "carry_chunks": ("training", "carry_chunks"),
    "bptt_chunks": ("training", "bptt_chunks"),
    "num_steps": ("training", "num_steps"),
    "learning_rate": ("training", "learning_rate"),
    "lr_schedule": ("training", "lr_schedule"),
    "min_lr": ("training", "min_lr"),
    "warmup_steps": ("training", "warmup_steps"),
    "lr_hold_steps": ("training", "lr_hold_steps"),
    "weight_decay": ("training", "weight_decay"),
    "hard_loss_gamma": ("training", "hard_loss_gamma"),
    "hard_loss_cap": ("training", "hard_loss_cap"),
    "grad_clip": ("training", "grad_clip"),
    "dropout": ("training", "dropout"),
    "amplifier_dtype": ("training", "amplifier_dtype"),
    "scan_backend": ("training", "scan_backend"),
    "gradient_checkpointing": ("training", "gradient_checkpointing"),
    "log_state_every": ("training", "log_state_every"),
    "full_val_final": ("training", "full_val_final"),
}


class ModelConfig:
    """Configuration backed by a model directory."""

    def __init__(self, model_dir: str | Path, data: dict[str, Any] | None = None):
        """Initialize a config wrapper.

        :param str | Path model_dir: Model directory for the configuration.
        :param dict[str, Any] | None data: Optional preloaded configuration data.
        """
        self.model_dir = Path(model_dir)
        self._data: dict[str, Any] = data or _deep_copy(DEFAULTS)

    # -- Paths within the model directory --

    @property
    def config_path(self) -> Path:
        """Return the config.json path.

        Returns:
            Path: The config.json path.
        """
        return self.model_dir / "config.json"

    @property
    def spec_path(self) -> Path:
        """Return the spec.pt path.

        Returns:
            Path: The spec.pt path.
        """
        return self.model_dir / "spec.pt"

    @property
    def tokenizer_path(self) -> Optional[Path]:
        """Return the first tokenizer model path in the model directory.

        Returns:
            Path | None: The first ``*.model`` file, if present.
        """
        candidates = list(self.model_dir.glob("*.model"))
        return candidates[0] if candidates else None

    @property
    def metrics_path(self) -> Path:
        """Return the metrics.jsonl path.

        Returns:
            Path: The metrics.jsonl path.
        """
        return self.model_dir / "metrics.jsonl"

    def checkpoint_path(self, step: Optional[int] = None) -> Path:
        """Return a checkpoint path for the requested step.

        :param Optional[int] step: Optional checkpoint step number.
        :return Path: The numbered checkpoint path or the default checkpoint.pt path.
        """
        if step is not None:
            return self.model_dir / f"checkpoint_{step}.pt"
        return self.model_dir / "checkpoint.pt"

    def latest_checkpoint(self) -> Optional[Path]:
        """Find the highest-numbered checkpoint.

        :return Path | None: The latest numbered checkpoint, final.pt, or checkpoint.pt.
        """
        numbered = sorted(self.model_dir.glob("checkpoint_*.pt"))
        if numbered:
            return numbered[-1]
        final = self.model_dir / "final.pt"
        if final.exists():
            return final
        generic = self.model_dir / "checkpoint.pt"
        return generic if generic.exists() else None

    # -- Section accessors --

    @property
    def model(self) -> dict[str, Any]:
        """Return the model section.

        Returns:
            dict[str, Any]: The ``model`` section.
        """
        return self._data.setdefault("model", {})

    @property
    def training(self) -> dict[str, Any]:
        """Return the training section.

        Returns:
            dict[str, Any]: The ``training`` section.
        """
        return self._data.setdefault("training", {})

    @property
    def data(self) -> dict[str, Any]:
        """Return the data section.

        Returns:
            dict[str, Any]: The ``data`` section.
        """
        return self._data.setdefault("data", {})

    @property
    def spec(self) -> dict[str, Any]:
        """Return the spec section.

        Returns:
            dict[str, Any]: The ``spec`` section.
        """
        return self._data.setdefault("spec", {})

    @property
    def meta(self) -> dict[str, Any]:
        """Return the meta section.

        Returns:
            dict[str, Any]: The ``meta`` section.
        """
        return self._data.setdefault("meta", {})

    # -- Convenience getters that flatten the hierarchy --

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Read a nested value from the loaded config.

        :param str section: Top-level config section.
        :param str key: Nested key within the section.
        :param Any default: Value to return when the key is missing.
        :return Any: The stored value or default.
        """
        return self._data.get(section, {}).get(key, default)

    @property
    def branch_lags_tuple(self) -> tuple[int, ...]:
        """Return branch lags normalized to a tuple of ints.

        Returns:
            tuple[int, ...]: Branch lags as a tuple of integers.
        """
        v = self.model.get("branch_lags", DEFAULTS["model"]["branch_lags"])
        if isinstance(v, str):
            return tuple(int(x) for x in v.split(",") if x)
        return tuple(int(x) for x in v)

    # -- Create / Load / Save --

    @classmethod
    def create(cls, model_dir: str | Path, **overrides: Any) -> "ModelConfig":
        """Create a new config with defaults.

        :param str | Path model_dir: Model directory to bind the config to.
        :param Any overrides: Optional config overrides keyed by CLI arg names.
        :return ModelConfig: A new config instance populated with defaults.
        """
        cfg = cls(model_dir, _deep_copy(DEFAULTS))
        cfg.meta["created"] = datetime.now(timezone.utc).isoformat()
        for key, value in overrides.items():
            if value is None:
                continue
            if key in _ARG_MAP:
                section, config_key = _ARG_MAP[key]
                cfg._data[section][config_key] = value
        return cfg

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelConfig":
        """Load config from an existing model directory.

        :param str | Path model_dir: Model directory to load from.
        :return ModelConfig: The loaded config instance.
        """
        p = Path(model_dir)
        config_path = p / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {p}")
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        # Merge with defaults for any missing keys
        merged = _deep_copy(DEFAULTS)
        _deep_update(merged, data)
        return cls(p, merged)

    @classmethod
    def load_or_create(cls, model_dir: str | Path, **overrides: Any) -> "ModelConfig":
        """Load existing config or create a new one.

        :param str | Path model_dir: Model directory to load or create.
        :param Any overrides: Optional config overrides keyed by CLI arg names.
        :return ModelConfig: The loaded or newly created config instance.
        """
        p = Path(model_dir)
        if (p / "config.json").exists():
            cfg = cls.load(p)
            cfg.update_from_dict(overrides)
            return cfg
        return cls.create(p, **overrides)

    def save(self) -> None:
        """Write config.json to the model directory."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=_json_default)

    # -- Update from CLI args --

    def update_from_args(self, args: Any) -> None:
        """Merge CLI args into config. Only non-None values override."""
        for arg_name, (section, key) in _ARG_MAP.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                self._data.setdefault(section, {})[key] = value

    def update_from_dict(self, d: dict[str, Any]) -> None:
        """Merge a flat dict of overrides using the arg map."""
        for key, value in d.items():
            if value is None:
                continue
            if key in _ARG_MAP:
                section, config_key = _ARG_MAP[key]
                self._data.setdefault(section, {})[config_key] = value

    # -- Tokenizer management --

    def find_or_copy_tokenizer(self, search_paths: Optional[list[Path]] = None) -> Optional[Path]:
        """Find or copy a tokenizer model.

        :param list[Path] | None search_paths: Optional extra roots to search first.
        :return Path | None: Existing or copied tokenizer path, or None if not found.
        """
        existing = self.tokenizer_path
        if existing is not None:
            return existing

        import glob

        roots = search_paths or []
        data_src = self.data.get("source")
        if data_src:
            dp = Path(data_src)
            roots.extend([dp, dp.parent, dp.parent / "tokenizers", dp.parent.parent / "tokenizers"])
        roots.extend([Path.cwd(), Path.cwd() / "data" / "tokenizers"])

        for root in roots:
            for pattern in ["*.model"]:
                for match in glob.glob(str(root / pattern)):
                    dest = self.model_dir / Path(match).name
                    if not dest.exists():
                        self.model_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(match, dest)
                    return dest
        return None

    # -- Display --

    def summary(self) -> str:
        """Render a human-readable config summary.

        :return str: A multi-line summary of the loaded config sections.
        """
        lines = [f"ModelConfig @ {self.model_dir}"]
        for section in ("model", "training", "data", "spec", "meta"):
            d = self._data.get(section, {})
            if d:
                lines.append(f"  [{section}]")
                for k, v in d.items():
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ModelConfig({self.model_dir})"


# -- Helpers --


def _deep_copy(d: dict) -> dict:
    """Deep-copy a JSON-compatible dict.

    Returns:
        dict: A deep-copied mapping.
    """
    return json.loads(json.dumps(d, default=_json_default))


def _deep_update(base: dict, override: dict) -> None:
    """Recursively merge override into base.

    :param dict base: Mapping to update in place.
    :param dict override: Mapping whose values overwrite ``base``.
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _json_default(obj: Any) -> Any:
    """Serialize paths and tuples for JSON output.

    Returns:
        Any: JSON-serializable fallback value.
    """
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


__all__ = ["ModelConfig", "DEFAULTS"]
