"""Analiza postojećih BER/AUC rezultata po performance grupama.

Ne radi novi embedding. Koristi:
- performance_groups.csv
- full_ber_results.csv
- steganalysis_features.csv

BER se samo ponovo agregira po performance_category.
AUC se ponovo evaluira Random Forest GroupKFold-om unutar performance grupa,
koristeći iste već generisane clean/stego feature-e.

Izlazi:
- performance_ber_summary.csv
- performance_auc_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from osu_stego.paths import (
    FULL_BER_RESULTS_DIR,
    METADATA_DIR,
    PERFORMANCE_RESULTS_DIR,
    STEGANALYSIS_RESULTS_DIR,
)

import pandas as pd

from scripts.experiments.run_steganalysis import evaluate_subset


DEFAULT_AUC_CONFIGS = [
    (10.0, 16),
    (15.0, 8),
    (15.0, 16),
    (20.0, 16),
]

PERFORMANCE_ORDER = [
    "Very Poor",
    "Poor",
    "Good",
    "Very Good",
]


def summarize_ber(
    full_ber: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    labels = performance[
        [
            "replay_file",
            "performance_category",
            "performance_percentile",
            "accuracy_percent",
        ]
    ].copy()

    merged = full_ber.merge(
        labels,
        on="replay_file",
        how="left",
        validate="many_to_one",
    )

    missing = merged[
        merged["performance_category"].isna()
    ]

    if not missing.empty:
        raise ValueError(
            f"{len(missing)} BER redova nema performance labelu."
        )

    ok = merged[
        merged["status"] == "OK"
    ].copy()

    summary = (
        ok.groupby(
            [
                "performance_category",
                "alpha",
                "n_frames_per_bit",
            ],
            observed=True,
        )
        .agg(
            replay_count=(
                "replay_file",
                "nunique",
            ),
            total_bits=(
                "message_bits",
                "sum",
            ),
            bit_errors=(
                "bit_errors_roundtrip",
                "sum",
            ),
            mean_ber_per_replay=(
                "ber_roundtrip",
                "mean",
            ),
            median_ber_per_replay=(
                "ber_roundtrip",
                "median",
            ),
            zero_ber_fraction=(
                "ber_roundtrip",
                lambda values: (
                    values == 0
                ).mean(),
            ),
            mean_active_carrier_fraction=(
                "active_carrier_fraction",
                "mean",
            ),
            total_positive_new_unmatched=(
                "positive_new_unmatched",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["ber"] = (
        summary["bit_errors"]
        / summary["total_bits"]
    )

    category_rank = {
        category: index
        for index, category
        in enumerate(PERFORMANCE_ORDER)
    }

    summary["_rank"] = (
        summary["performance_category"]
        .map(category_rank)
    )

    return (
        summary.sort_values(
            [
                "alpha",
                "n_frames_per_bit",
                "_rank",
            ]
        )
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def summarize_auc(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    labels = performance[
        [
            "replay_file",
            "performance_category",
        ]
    ].copy()

    merged = features.merge(
        labels,
        on="replay_file",
        how="left",
        validate="many_to_one",
    )

    if merged[
        "performance_category"
    ].isna().any():
        raise ValueError(
            "Neki stegoanalysis feature redovi nemaju performance labelu."
        )

    # evaluate_subset filtrira kolonu "category".
    # Ovde je namerno preusmeravamo na novu performance labelu.
    merged = merged.copy()
    merged["category"] = (
        merged["performance_category"]
    )

    rows: list[dict] = []

    available_configs = set(
        zip(
            merged["alpha"].astype(float),
            merged["n_frames_per_bit"].astype(int),
        )
    )

    for alpha, n_value in DEFAULT_AUC_CONFIGS:
        if (alpha, n_value) not in available_configs:
            continue

        for category in PERFORMANCE_ORDER:
            summary, _ = evaluate_subset(
                frame=merged,
                scope=category,
                alpha=alpha,
                n_value=n_value,
                folds=folds,
                trees=trees,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
            )

            summary[
                "performance_category"
            ] = category

            rows.append(summary)

    result = pd.DataFrame(rows)

    category_rank = {
        category: index
        for index, category
        in enumerate(PERFORMANCE_ORDER)
    }

    result["_rank"] = (
        result["performance_category"]
        .map(category_rank)
    )

    return (
        result.sort_values(
            [
                "alpha",
                "n_frames_per_bit",
                "_rank",
            ]
        )
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--performance",
        type=Path,
        default=METADATA_DIR / "performance_groups.csv",
    )
    parser.add_argument(
        "--ber-results",
        type=Path,
        default=FULL_BER_RESULTS_DIR / "full_ber_results.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=STEGANALYSIS_RESULTS_DIR / "steganalysis_features.csv",
    )
    parser.add_argument(
        "--ber-output",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR / "performance_ber_summary.csv",
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR / "performance_auc_summary.csv",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    performance = pd.read_csv(
        args.performance
    )
    full_ber = pd.read_csv(
        args.ber_results
    )
    features = pd.read_csv(
        args.features
    )

    ber_summary = summarize_ber(
        full_ber=full_ber,
        performance=performance,
    )

    ber_summary.to_csv(
        args.ber_output,
        index=False,
    )

    print(
        f"BER summary: {args.ber_output}"
    )

    auc_summary = summarize_auc(
        features=features,
        performance=performance,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=(
            args.bootstrap_iterations
        ),
    )

    auc_summary.to_csv(
        args.auc_output,
        index=False,
    )

    print(
        f"AUC summary: {args.auc_output}"
    )

    print("\nGlavna konfiguracija alpha=10, N=16:")

    selected_ber = ber_summary[
        (ber_summary["alpha"] == 10)
        & (
            ber_summary[
                "n_frames_per_bit"
            ] == 16
        )
    ]

    print("\nBER:")
    print(
        selected_ber[
            [
                "performance_category",
                "replay_count",
                "ber",
                "zero_ber_fraction",
            ]
        ].to_string(index=False)
    )

    selected_auc = auc_summary[
        (auc_summary["alpha"] == 10)
        & (
            auc_summary[
                "n_frames_per_bit"
            ] == 16
        )
    ]

    print("\nAUC:")
    print(
        selected_auc[
            [
                "performance_category",
                "replay_pairs",
                "roc_auc",
                "auc_ci95_low",
                "auc_ci95_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
