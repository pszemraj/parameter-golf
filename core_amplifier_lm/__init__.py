"""Core/Amplifier Language Model.

A hybrid frozen-statistics + tiny-learnable-recurrent language model.
"""

from .config import DEFAULTS, ModelConfig, trigram_memory_config_value
from .model import (
    AmplifierSpec,
    CoreAmplifierLM,
    build_amplifier_spec,
    default_16mb_recipe,
    estimate_storage_bytes,
)
from .spec_builder import (
    add_trigram_memory_to_spec,
    build_spec_optimized,
    count_all,
    load_tokens_int32,
    load_train_val_int32,
    training_token_file_fingerprint,
)
from .trigram_memory import (
    spec_with_trigram_memory_table,
    trigram_memory_table_from_spec,
    validate_trigram_memory_table,
)

__all__ = [
    "AmplifierSpec",
    "CoreAmplifierLM",
    "DEFAULTS",
    "ModelConfig",
    "build_amplifier_spec",
    "add_trigram_memory_to_spec",
    "build_spec_optimized",
    "count_all",
    "default_16mb_recipe",
    "estimate_storage_bytes",
    "load_tokens_int32",
    "load_train_val_int32",
    "trigram_memory_config_value",
    "spec_with_trigram_memory_table",
    "trigram_memory_table_from_spec",
    "training_token_file_fingerprint",
    "validate_trigram_memory_table",
]
