"""Ordered HSV reading strategy: strict current method, then historical fallbacks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from direct_hsv_reader import DirectFlapAnalysis, analyze_direct_flap_strip
from fixed_center_reader import analyze_fixed_center_strip
from legacy_candidate_reader import analyze_legacy_candidate_strips


METHODS: tuple[tuple[str, Callable[[np.ndarray], DirectFlapAnalysis]], ...] = (
    ("dense_bottom_connected", analyze_direct_flap_strip),
    ("legacy_76_candidate_change_peak", analyze_legacy_candidate_strips),
    ("fixed_center_24_percent", analyze_fixed_center_strip),
)


def analyze_flap_strip_with_fallback(image: np.ndarray) -> DirectFlapAnalysis:
    failures: list[dict[str, str]] = []
    for method_index, (method_name, method) in enumerate(METHODS, start=1):
        try:
            result = method(image)
        except ValueError as exc:
            failures.append({"method": method_name, "error": str(exc)})
            continue
        result.metrics = {
            **result.metrics,
            "selected_method": method_name,
            "selected_method_index": method_index,
            "fallback_used": method_index > 1,
            "previous_method_failures": failures,
            "method_order": [name for name, _ in METHODS],
        }
        return result
    detail = "；".join(
        f"{failure['method']}: {failure['error']}" for failure in failures
    )
    raise ValueError(f"所有 HSV 读数方法均失败。{detail}")

