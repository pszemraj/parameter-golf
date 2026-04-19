"""Trainer logging helper tests."""

from __future__ import annotations

from train_core_amplifier import _format_debug_vector


def test_format_debug_vector_keeps_all_values():
    values = [float(i) for i in range(12)]
    formatted = _format_debug_vector(values)
    assert formatted.count(",") == 11
    assert formatted.startswith("0.000,1.000,2.000")
    assert formatted.endswith("9.000,10.000,11.000")


def test_format_debug_vector_handles_empty_values():
    assert _format_debug_vector([]) == "none"
