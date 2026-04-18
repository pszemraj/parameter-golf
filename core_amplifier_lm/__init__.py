"""Core/Amplifier Language Model.

A hybrid frozen-statistics + tiny-learnable-recurrent language model.
"""

from .config import DEFAULTS, ModelConfig
from .model import (
    AmplifierSpec,
    CoreAmplifierLM,
    build_amplifier_spec,
    default_16mb_recipe,
    estimate_storage_bytes,
)
from .spec_builder import (
    build_spec_optimized,
    count_all,
    load_tokens_int32,
    load_train_val_int32,
)

__all__ = [
    "AmplifierSpec",
    "CoreAmplifierLM",
    "DEFAULTS",
    "ModelConfig",
    "build_amplifier_spec",
    "build_spec_optimized",
    "count_all",
    "default_16mb_recipe",
    "estimate_storage_bytes",
    "load_tokens_int32",
    "load_train_val_int32",
]
