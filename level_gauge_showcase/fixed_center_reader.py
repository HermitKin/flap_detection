"""Historical fixed-centre 24% strip reader, retained as the final fallback."""

from __future__ import annotations

import numpy as np

from direct_hsv_reader import DirectFlapAnalysis
from legacy_candidate_reader import (
    _peak_z_score,
    extract_legacy_row_features,
    find_persistent_change_peak,
)


def analyze_fixed_center_strip(
    image: np.ndarray, width_ratio: float = 0.24
) -> DirectFlapAnalysis:
    height, width = image.shape[:2]
    if height < 40 or width < 7:
        raise ValueError(f"固定中央条带法输入过小：{width}×{height}。")
    strip_width = max(7, int(round(width * width_ratio)))
    x1 = max(0, (width - strip_width) // 2)
    x2 = min(width, x1 + strip_width)
    features, names = extract_legacy_row_features(image, x1, x2)
    level_y, score = find_persistent_change_peak(features)
    peak_z = _peak_z_score(score, level_y)
    if peak_z <= 0:
        raise ValueError("固定中央条带没有找到有效变化峰。")
    selected_columns = np.zeros(width, dtype=bool)
    selected_columns[x1:x2] = True
    return DirectFlapAnalysis(
        x1=x1,
        x2=x2,
        features=features,
        feature_names=names,
        level_y=level_y,
        bottom_state_score=score.astype(np.float32),
        bottom_connected_mask=np.arange(height) >= level_y,
        localization_profile=np.zeros(width, dtype=np.float32),
        localization_mask=selected_columns,
        confidence=float(peak_z),
        metrics={
            "method": "fixed_center_24_percent_change_peak",
            "boundary_method": "persistent_multivariate_change_peak",
            "localization": {
                "method": "fixed_center_strip",
                "selected_x1": x1,
                "selected_x2": x2,
                "selected_width": x2 - x1,
                "selected_width_ratio": round((x2 - x1) / width, 6),
                "localization_confidence": round(float(peak_z), 6),
                "fixed_center_used": True,
                "candidate_grid_used": False,
            },
            "peak_z": round(float(peak_z), 6),
        },
    )

