"""Controlled, note-index-aware timing features for steganalysis.

The extractor accepts only a note-indexed residual array. Missing matches remain
as NaN at their original note positions. No message, PN key, label, replay
population statistic, or decoder output is accepted by this API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


FEATURE_VERSION = "strong-steganalysis-features-v1"

BASELINE_FEATURES = (
    "variance",
    "excess_kurtosis",
    "autocorrelation_lag1",
)

GLOBAL_FEATURES = (
    "skewness",
    "iqr",
    "mean_absolute_residual",
    "quantile_10",
    "quantile_25",
    "quantile_50",
    "quantile_75",
    "quantile_90",
)

TEMPORAL_FEATURES = (
    "autocorrelation_lag2",
    "autocorrelation_lag4",
    "autocorrelation_lag8",
    "mean_absolute_first_difference",
    "variance_first_difference",
    "sign_change_rate",
)

POSITION_FEATURES = tuple(
    f"quarter_{quarter}_{statistic}"
    for quarter in range(1, 5)
    for statistic in (
        "variance",
        "autocorrelation_lag1",
        "mean_absolute_residual",
    )
)

FULL_FEATURES = (
    BASELINE_FEATURES
    + GLOBAL_FEATURES
    + TEMPORAL_FEATURES
    + POSITION_FEATURES
)

FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "baseline": BASELINE_FEATURES,
    "baseline_global": BASELINE_FEATURES + GLOBAL_FEATURES,
    "baseline_temporal": BASELINE_FEATURES + TEMPORAL_FEATURES,
    "baseline_position": BASELINE_FEATURES + POSITION_FEATURES,
    "full": FULL_FEATURES,
}


def _finite_or_zero(value: float) -> float:
    result = float(value)
    return result if np.isfinite(result) else 0.0


def _distribution_statistics(values: np.ndarray) -> dict[str, float]:
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return {
            "variance": 0.0,
            "excess_kurtosis": 0.0,
            "skewness": 0.0,
            "iqr": 0.0,
            "mean_absolute_residual": 0.0,
            "quantile_10": 0.0,
            "quantile_25": 0.0,
            "quantile_50": 0.0,
            "quantile_75": 0.0,
            "quantile_90": 0.0,
        }

    mean = float(np.mean(valid_values))
    centered = valid_values - mean
    variance = float(np.mean(centered**2))
    if variance > 0.0:
        excess_kurtosis = float(np.mean(centered**4) / variance**2 - 3.0)
        skewness = float(np.mean(centered**3) / variance**1.5)
    else:
        excess_kurtosis = 0.0
        skewness = 0.0

    quantiles = np.quantile(valid_values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "variance": _finite_or_zero(variance),
        "excess_kurtosis": _finite_or_zero(excess_kurtosis),
        "skewness": _finite_or_zero(skewness),
        "iqr": _finite_or_zero(float(quantiles[3] - quantiles[1])),
        "mean_absolute_residual": _finite_or_zero(
            float(np.mean(np.abs(valid_values)))
        ),
        "quantile_10": _finite_or_zero(float(quantiles[0])),
        "quantile_25": _finite_or_zero(float(quantiles[1])),
        "quantile_50": _finite_or_zero(float(quantiles[2])),
        "quantile_75": _finite_or_zero(float(quantiles[3])),
        "quantile_90": _finite_or_zero(float(quantiles[4])),
    }


def _indexed_autocorrelation(values: np.ndarray, lag: int) -> float:
    """Legacy-compatible autocorrelation without compressing missing notes."""
    if lag <= 0:
        raise ValueError("Autocorrelation lag mora biti > 0.")
    valid = ~np.isnan(values)
    valid_values = values[valid]
    if len(valid_values) == 0 or len(values) <= lag:
        return 0.0
    mean = float(np.mean(valid_values))
    centered = values - mean
    denominator = float(np.sum((valid_values - mean) ** 2))
    adjacent = valid[:-lag] & valid[lag:]
    if denominator <= 0.0 or not np.any(adjacent):
        return 0.0
    numerator = float(
        np.sum(centered[:-lag][adjacent] * centered[lag:][adjacent])
    )
    return _finite_or_zero(numerator / denominator)


def _temporal_statistics(values: np.ndarray) -> dict[str, float]:
    valid = ~np.isnan(values)
    adjacent = valid[:-1] & valid[1:]
    differences = values[1:][adjacent] - values[:-1][adjacent]
    sign_changes = (
        (values[1:][adjacent] * values[:-1][adjacent]) < 0.0
    )
    return {
        "autocorrelation_lag1": _indexed_autocorrelation(values, 1),
        "autocorrelation_lag2": _indexed_autocorrelation(values, 2),
        "autocorrelation_lag4": _indexed_autocorrelation(values, 4),
        "autocorrelation_lag8": _indexed_autocorrelation(values, 8),
        "mean_absolute_first_difference": (
            _finite_or_zero(float(np.mean(np.abs(differences))))
            if len(differences)
            else 0.0
        ),
        "variance_first_difference": (
            _finite_or_zero(float(np.var(differences)))
            if len(differences)
            else 0.0
        ),
        "sign_change_rate": (
            _finite_or_zero(float(np.mean(sign_changes)))
            if len(sign_changes)
            else 0.0
        ),
    }


def _position_statistics(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    # np.array_split partitions the original note-index axis; NaNs are not
    # removed until statistics are calculated inside each fixed quarter.
    for quarter, indices in enumerate(
        np.array_split(np.arange(len(values), dtype=np.int64), 4),
        1,
    ):
        quarter_values = values[indices]
        distribution = _distribution_statistics(quarter_values)
        result[f"quarter_{quarter}_variance"] = distribution["variance"]
        result[f"quarter_{quarter}_autocorrelation_lag1"] = (
            _indexed_autocorrelation(quarter_values, 1)
        )
        result[f"quarter_{quarter}_mean_absolute_residual"] = distribution[
            "mean_absolute_residual"
        ]
    return result


def residual_timing_features(
    residuals: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Extract the frozen v1 feature schema from note-indexed residuals."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Residuali moraju biti jednodimenzionalni.")
    if np.any(np.isinf(values)):
        raise ValueError("Residuali ne smeju sadržati +/-inf.")

    distribution = _distribution_statistics(values)
    temporal = _temporal_statistics(values)
    result: dict[str, float | int] = {
        **distribution,
        **temporal,
        **_position_statistics(values),
        "num_notes": int(len(values)),
        "n_valid_residuals": int(np.sum(~np.isnan(values))),
        "valid_residual_fraction": (
            float(np.mean(~np.isnan(values))) if len(values) else 0.0
        ),
    }
    if tuple(name for name in FULL_FEATURES if name not in result):
        raise RuntimeError("Interna greška: feature schema nije kompletna.")
    if not np.all(
        np.isfinite(np.asarray([result[name] for name in FULL_FEATURES], dtype=float))
    ):
        raise RuntimeError("Extractor je proizveo NaN/inf karakteristiku.")
    return result
