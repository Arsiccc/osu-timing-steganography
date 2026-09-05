"""Targeted capacity-candidate analysis.

Run from project root:

    python3 -m scripts.analysis.analyze_capacity_candidates

Candidates:
    alpha=20, N=4
    alpha=15, N=6
    alpha=20, N=6
    alpha=15, N=8   (existing reference)

The script:
- generates missing stegoanalysis feature pairs only for these configs
- computes overall AUC into separate files
- computes BER and AUC by performance group
- writes a compact capacity/BER/AUC trade-off table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from osu_stego.paths import (
    CONFIG_DIR,
    DATASET_DIR,
    FULL_BER_RESULTS_DIR,
    METADATA_DIR,
    PERFORMANCE_RESULTS_DIR,
    STEGANALYSIS_RESULTS_DIR,
)
from scripts.experiments.run_steganalysis import (
    evaluate_subset,
    generate_features,
)


CANDIDATES = [
    (20.0, 4),
    (15.0, 6),
    (20.0, 6),
    (15.0, 8),
]

PERFORMANCE_ORDER = [
    "Very Poor",
    "Poor",
    "Good",
    "Very Good",
]


def is_candidate(row: pd.Series) -> bool:
    return (
        float(row["alpha"]),
        int(row["n_frames_per_bit"]),
    ) in set(CANDIDATES)


def summarize_ber_by_performance(
    full_ber: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    labels = performance[
        [
            "replay_file",
            "performance_category",
            "accuracy_percent",
            "performance_percentile",
        ]
    ].copy()

    merged = full_ber.merge(
        labels,
        on="replay_file",
        how="left",
        validate="many_to_one",
    )

    if merged["performance_category"].isna().any():
        missing = int(
            merged["performance_category"].isna().sum()
        )
        raise ValueError(
            f"{missing} BER redova nema performance labelu."
        )

    candidate_set = set(CANDIDATES)
    mask = [
        (
            float(alpha),
            int(n_value),
        )
        in candidate_set
        for alpha, n_value in zip(
            merged["alpha"],
            merged["n_frames_per_bit"],
        )
    ]

    selected = merged[
        pd.Series(mask, index=merged.index)
        & (merged["status"] == "OK")
    ].copy()

    summary = (
        selected.groupby(
            [
                "performance_category",
                "alpha",
                "n_frames_per_bit",
            ],
            observed=True,
        )
        .agg(
            replay_count=("replay_file", "nunique"),
            total_bits=("message_bits", "sum"),
            bit_errors=("bit_errors_roundtrip", "sum"),
            mean_ber_per_replay=("ber_roundtrip", "mean"),
            median_ber_per_replay=("ber_roundtrip", "median"),
            zero_ber_fraction=(
                "ber_roundtrip",
                lambda values: (values == 0).mean(),
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
        summary["bit_errors"] / summary["total_bits"]
    )
    summary["capacity_multiplier_vs_n16"] = (
        16.0 / summary["n_frames_per_bit"]
    )

    rank = {
        category: index
        for index, category in enumerate(PERFORMANCE_ORDER)
    }
    summary["_rank"] = (
        summary["performance_category"].map(rank)
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


def evaluate_overall_auc(
    features: pd.DataFrame,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for alpha, n_value in CANDIDATES:
        summary, _ = evaluate_subset(
            frame=features,
            scope="ALL",
            alpha=alpha,
            n_value=n_value,
            folds=folds,
            trees=trees,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
        )
        summary["capacity_multiplier_vs_n16"] = 16.0 / n_value
        rows.append(summary)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "n_frames_per_bit",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )


def evaluate_auc_by_performance(
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

    if merged["performance_category"].isna().any():
        raise ValueError(
            "Neki feature redovi nemaju performance labelu."
        )

    merged = merged.copy()
    merged["category"] = merged["performance_category"]

    rows: list[dict] = []

    for alpha, n_value in CANDIDATES:
        for performance_category in PERFORMANCE_ORDER:
            summary, _ = evaluate_subset(
                frame=merged,
                scope=performance_category,
                alpha=alpha,
                n_value=n_value,
                folds=folds,
                trees=trees,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
            )
            summary["performance_category"] = performance_category
            summary["capacity_multiplier_vs_n16"] = 16.0 / n_value
            rows.append(summary)

    result = pd.DataFrame(rows)

    rank = {
        category: index
        for index, category in enumerate(PERFORMANCE_ORDER)
    }
    result["_rank"] = (
        result["performance_category"].map(rank)
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


def build_tradeoff_table(
    overall_auc: pd.DataFrame,
    full_ber: pd.DataFrame,
) -> pd.DataFrame:
    candidate_set = set(CANDIDATES)

    mask = [
        (
            float(alpha),
            int(n_value),
        )
        in candidate_set
        for alpha, n_value in zip(
            full_ber["alpha"],
            full_ber["n_frames_per_bit"],
        )
    ]

    selected = full_ber[
        pd.Series(mask, index=full_ber.index)
        & (full_ber["status"] == "OK")
    ].copy()

    ber_overall = (
        selected.groupby(
            [
                "alpha",
                "n_frames_per_bit",
            ],
            as_index=False,
        )
        .agg(
            total_bits=("message_bits", "sum"),
            bit_errors=("bit_errors_roundtrip", "sum"),
            replay_experiments=("replay_file", "size"),
            mean_active_carrier_fraction=(
                "active_carrier_fraction",
                "mean",
            ),
            total_positive_new_unmatched=(
                "positive_new_unmatched",
                "sum",
            ),
        )
    )

    ber_overall["ber"] = (
        ber_overall["bit_errors"] / ber_overall["total_bits"]
    )

    auc_columns = [
        "alpha",
        "n_frames_per_bit",
        "roc_auc",
        "auc_ci95_low",
        "auc_ci95_high",
        "capacity_multiplier_vs_n16",
    ]

    result = ber_overall.merge(
        overall_auc[auc_columns],
        on=[
            "alpha",
            "n_frames_per_bit",
        ],
        how="inner",
        validate="one_to_one",
    )

    result["ber_under_5pct"] = result["ber"] < 0.05

    return result.sort_values(
        [
            "capacity_multiplier_vs_n16",
            "roc_auc",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted BER/AUC analysis for high-capacity configs."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=METADATA_DIR / "results_v3_clean.csv",
    )
    parser.add_argument(
        "--map-offsets",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
    )
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
        "--overall-auc-output",
        type=Path,
        default=STEGANALYSIS_RESULTS_DIR / "capacity_candidates_auc.csv",
    )
    parser.add_argument(
        "--performance-auc-output",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR
        / "capacity_candidates_auc_by_performance.csv",
    )
    parser.add_argument(
        "--performance-ber-output",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR
        / "capacity_candidates_ber_by_performance.csv",
    )
    parser.add_argument(
        "--tradeoff-output",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR
        / "capacity_candidates_tradeoff.csv",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--key",
        type=str,
        default="pilot-ber-key-v1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--hit-margin-ms",
        type=float,
        default=5.0,
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
        "--bootstrap-iterations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
    )

    args = parser.parse_args()

    for path in (
        args.overall_auc_output,
        args.performance_auc_output,
        args.performance_ber_output,
        args.tradeoff_output,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if not args.evaluate_only:
        results = pd.read_csv(
            args.results
        )

        print("=" * 100)
        print("GENERISANJE TARGETED STEGOANALYSIS FEATURE-A")
        print("=" * 100)
        print("Kandidati:", CANDIDATES)
        print()

        generate_features(
            results=results,
            dataset_dir=args.dataset_dir,
            map_offsets_path=args.map_offsets,
            feature_path=args.features,
            configs=CANDIDATES,
            bits=args.bits,
            key=args.key,
            seed=args.seed,
            hit_margin_ms=args.hit_margin_ms,
            overwrite=False,
            limit_per_category=None,
        )

    features = pd.read_csv(
        args.features
    )
    performance = pd.read_csv(
        args.performance
    )
    full_ber = pd.read_csv(
        args.ber_results
    )

    overall_auc = evaluate_overall_auc(
        features=features,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    overall_auc.to_csv(
        args.overall_auc_output,
        index=False,
    )

    performance_ber = summarize_ber_by_performance(
        full_ber=full_ber,
        performance=performance,
    )
    performance_ber.to_csv(
        args.performance_ber_output,
        index=False,
    )

    performance_auc = evaluate_auc_by_performance(
        features=features,
        performance=performance,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    performance_auc.to_csv(
        args.performance_auc_output,
        index=False,
    )

    tradeoff = build_tradeoff_table(
        overall_auc=overall_auc,
        full_ber=full_ber,
    )
    tradeoff.to_csv(
        args.tradeoff_output,
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL CANDIDATE TRADE-OFF")
    print("=" * 100)
    print(
        tradeoff[
            [
                "alpha",
                "n_frames_per_bit",
                "capacity_multiplier_vs_n16",
                "ber",
                "roc_auc",
                "auc_ci95_low",
                "auc_ci95_high",
                "mean_active_carrier_fraction",
                "total_positive_new_unmatched",
            ]
        ].to_string(index=False)
    )

    print("\nFajlovi:")
    print(f"  {args.overall_auc_output}")
    print(f"  {args.performance_ber_output}")
    print(f"  {args.performance_auc_output}")
    print(f"  {args.tradeoff_output}")


if __name__ == "__main__":
    main()
