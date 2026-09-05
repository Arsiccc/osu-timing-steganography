"""Paired fixed-alpha vs performance-adaptive alpha experiment.

The adaptive policy uses only the precomputed map-relative performance group
derived from official replay hit counts. It never reads timing residual
statistics when selecting alpha.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from scripts.experiments.run_layout_comparison import (
    FEATURE_FIELDS as LAYOUT_FEATURE_FIELDS,
    RESULT_FIELDS as LAYOUT_RESULT_FIELDS,
    _read_rows_with_schema,
    bootstrap_auc_delta,
    bootstrap_weighted_ber_delta,
    experiment_config_id,
    key_id,
    replace_configs_rows,
    run_one,
)
from scripts.experiments.run_payload_sweep import (
    deterministic_message,
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    prepare_replay,
    stable_seed,
)
from scripts.experiments.run_steganalysis import evaluate_subset


EXPERIMENT_VERSION = "performance-adaptive-alpha-v1"
POLICY_ID = "map-relative-official-accuracy-v1"
PN_KEY = "adaptive-alpha-pn-v1"
PREFIX_LAYOUT_KEY = "prefix-layout-unused-v1"
N_VALUE = 8
PAYLOAD_FRACTION = 0.075
CV_GROUP_COLUMNS = ("replay_file", "beatmap_hash")

PERFORMANCE_ORDER = (
    "Very Poor",
    "Poor",
    "Good",
    "Very Good",
)
ADAPTIVE_ALPHA = {
    "Very Poor": 20.0,
    "Poor": 16.0,
    "Good": 12.0,
    "Very Good": 10.0,
}
POLICY_DEFINITION = ";".join(
    f"{category}={ADAPTIVE_ALPHA[category]:g}"
    for category in PERFORMANCE_ORDER
)
METHOD_ALPHA = {
    "fixed_14": 14.0,
    "fixed_15": 15.0,
}
METHODS = (*METHOD_ALPHA, "adaptive_v1")

EXTRA_RESULT_FIELDS = [
    "method",
    "policy_id",
    "policy_definition",
    "performance_category",
]
RESULT_FIELDS = (
    LAYOUT_RESULT_FIELDS[:2]
    + EXTRA_RESULT_FIELDS
    + LAYOUT_RESULT_FIELDS[2:]
)
FEATURE_FIELDS = EXTRA_RESULT_FIELDS + LAYOUT_FEATURE_FIELDS


def completed_result_keys(path: Path) -> set[str]:
    frame = pd.DataFrame(_read_rows_with_schema(path, RESULT_FIELDS))
    if frame.empty:
        return set()
    return set(
        frame.loc[frame["status"] == "OK", "config_id"].astype(str)
    )


def completed_feature_keys(path: Path) -> set[str]:
    frame = pd.DataFrame(_read_rows_with_schema(path, FEATURE_FIELDS))
    if frame.empty:
        return set()
    completed: set[str] = set()
    for config_id, group in frame.groupby("config_id"):
        if tuple(sorted(group["label"].astype(int).tolist())) == (0, 1):
            completed.add(str(config_id))
    return completed


def alpha_for_method(method: str, performance_category: str) -> float:
    if method in METHOD_ALPHA:
        return METHOD_ALPHA[method]
    if method == "adaptive_v1":
        try:
            return ADAPTIVE_ALPHA[performance_category]
        except KeyError as exc:
            raise ValueError(
                f"Nepoznata performance kategorija: {performance_category}."
            ) from exc
    raise ValueError(f"Nepoznat metod: {method}.")


def load_experiment_cohort(
    results_path: Path,
    performance_path: Path,
    per_group_limit: int | None,
) -> pd.DataFrame:
    source = pd.read_csv(results_path)
    performance = pd.read_csv(performance_path)[
        ["replay_file", "beatmap_hash", "performance_category"]
    ]

    if performance["replay_file"].duplicated().any():
        raise ValueError("Performance metadata ima duplirana replay imena.")

    cohort = source.merge(
        performance,
        on=["replay_file", "beatmap_hash"],
        how="inner",
        validate="one_to_one",
    )
    if len(cohort) != len(source):
        raise ValueError(
            f"Performance metadata pokriva {len(cohort)}/{len(source)} replay-eva."
        )
    if not set(cohort["performance_category"]).issubset(PERFORMANCE_ORDER):
        raise ValueError("Cohort sadrži nepoznatu performance kategoriju.")

    if per_group_limit is not None:
        if per_group_limit <= 0:
            raise ValueError("--per-group-limit mora biti > 0.")
        cohort = pd.concat(
            [
                cohort[
                    cohort["performance_category"] == category
                ].head(per_group_limit)
                for category in PERFORMANCE_ORDER
            ],
            ignore_index=True,
        )

    return cohort


def validate_complete_rows(
    results: pd.DataFrame,
    features: pd.DataFrame,
    expected_configs: int,
) -> None:
    if len(results) != expected_configs:
        raise RuntimeError(
            f"Očekivano {expected_configs} result redova, dobijeno {len(results)}."
        )
    if results["config_id"].duplicated().any():
        raise ValueError("Dupliran config_id u result CSV-u.")
    if set(results["status"]) != {"OK"}:
        raise RuntimeError("Nisu sve adaptive-alpha konfiguracije uspešne.")

    labels = features.groupby("config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if len(labels) != expected_configs or not all(
        value == (0, 1) for value in labels.tolist()
    ):
        raise ValueError("Svaka konfiguracija mora imati jedan clean/stego par.")

    counts = results.groupby("replay_file")["method"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    expected_methods = tuple(sorted(METHODS))
    if not all(value == expected_methods for value in counts.tolist()):
        raise ValueError("Replay nema tačno jednu instancu svakog metoda.")


def _paired_auc_seed(
    seed: int,
    control_method: str,
    cv_group_column: str,
) -> int:
    return stable_seed(
        seed,
        EXPERIMENT_VERSION,
        control_method,
        cv_group_column,
        "paired-auc",
    )


def evaluate_methods(
    results: pd.DataFrame,
    features: pd.DataFrame,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_rows: list[dict] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}

    for cv_group_column in CV_GROUP_COLUMNS:
        for method in METHODS:
            subset = features[features["method"] == method].copy()
            summary, prediction = evaluate_subset(
                frame=subset,
                scope="ALL",
                alpha=None,
                n_value=N_VALUE,
                folds=folds,
                trees=trees,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
                cv_group_column=cv_group_column,
            )
            applied = subset[
                subset["label"] == 1
            ]["alpha"].to_numpy(dtype=np.float64)
            summary.update(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "policy_id": POLICY_ID,
                    "policy_definition": POLICY_DEFINITION,
                    "method": method,
                    "cv_group_column": cv_group_column,
                    "mean_applied_alpha": float(np.mean(applied)),
                    "rms_applied_alpha": float(np.sqrt(np.mean(applied**2))),
                    "classifier_trees": trees,
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )
            auc_rows.append(summary)
            predictions[(cv_group_column, method)] = prediction

    paired_rows: list[dict] = []
    adaptive = results[results["method"] == "adaptive_v1"]

    for control_method in METHOD_ALPHA:
        control = results[results["method"] == control_method]
        paired = control[
            ["replay_file", "message_bits", "bit_errors_roundtrip"]
        ].rename(
            columns={"bit_errors_roundtrip": "errors_prefix"}
        ).merge(
            adaptive[
                ["replay_file", "message_bits", "bit_errors_roundtrip"]
            ].rename(
                columns={
                    "message_bits": "message_bits_adaptive",
                    "bit_errors_roundtrip": "errors_distributed",
                }
            ),
            on="replay_file",
            validate="one_to_one",
        )
        if not np.array_equal(
            paired["message_bits"],
            paired["message_bits_adaptive"],
        ):
            raise ValueError("Metodi nemaju isti payload po replay-u.")

        total_bits = int(paired["message_bits"].sum())
        ber_control = float(paired["errors_prefix"].sum() / total_bits)
        ber_adaptive = float(paired["errors_distributed"].sum() / total_bits)
        ber_low, ber_high = bootstrap_weighted_ber_delta(
            paired,
            stable_seed(
                seed,
                EXPERIMENT_VERSION,
                control_method,
                "paired-ber",
            ),
            bootstrap_iterations,
        )

        for cv_group_column in CV_GROUP_COLUMNS:
            control_auc = next(
                row for row in auc_rows
                if row["method"] == control_method
                and row["cv_group_column"] == cv_group_column
            )
            adaptive_auc = next(
                row for row in auc_rows
                if row["method"] == "adaptive_v1"
                and row["cv_group_column"] == cv_group_column
            )
            auc_low, auc_high = bootstrap_auc_delta(
                predictions[(cv_group_column, control_method)],
                predictions[(cv_group_column, "adaptive_v1")],
                _paired_auc_seed(seed, control_method, cv_group_column),
                bootstrap_iterations,
            )
            paired_rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "policy_id": POLICY_ID,
                    "policy_definition": POLICY_DEFINITION,
                    "control_method": control_method,
                    "cv_group_column": cv_group_column,
                    "cv_groups": int(control_auc["cv_groups"]),
                    "replay_pairs": len(paired),
                    "total_bits": total_bits,
                    "ber_control": ber_control,
                    "ber_adaptive": ber_adaptive,
                    "delta_ber_adaptive_minus_control": (
                        ber_adaptive - ber_control
                    ),
                    "delta_ber_ci95_low": ber_low,
                    "delta_ber_ci95_high": ber_high,
                    "auc_control": control_auc["roc_auc"],
                    "auc_adaptive": adaptive_auc["roc_auc"],
                    "delta_auc_adaptive_minus_control": (
                        adaptive_auc["roc_auc"] - control_auc["roc_auc"]
                    ),
                    "delta_auc_ci95_low": auc_low,
                    "delta_auc_ci95_high": auc_high,
                    "folds": folds,
                    "classifier_trees": trees,
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )

    return pd.DataFrame(auc_rows), pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed and performance-adaptive alpha."
    )
    output_dir = RESULTS_DIR / "adaptive_alpha"
    parser.add_argument(
        "--results",
        type=Path,
        default=METADATA_DIR / "results_v3_clean.csv",
    )
    parser.add_argument(
        "--performance-groups",
        type=Path,
        default=METADATA_DIR / "performance_groups.csv",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--map-offsets",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
    )
    parser.add_argument("--per-group-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hit-margin-ms", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=output_dir / "adaptive_alpha_results.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=output_dir / "adaptive_alpha_features.csv",
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=output_dir / "adaptive_alpha_auc.csv",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=output_dir / "adaptive_alpha_paired.csv",
    )
    args = parser.parse_args()

    cohort = load_experiment_cohort(
        args.results,
        args.performance_groups,
        args.per_group_limit,
    )
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    completed = completed_result_keys(args.output) & completed_feature_keys(
        args.features
    )
    expected_configs = len(cohort) * len(METHODS)
    new_ok = 0

    print("=" * 100)
    print("PERFORMANCE-ADAPTIVE ALPHA")
    print("=" * 100)
    print(f"Replay-eva: {len(cohort)}")
    print(f"Metodi:     {METHODS}")
    print(f"Policy:     {ADAPTIVE_ALPHA}")
    print(f"N/payload:  {N_VALUE}/{PAYLOAD_FRACTION:.1%}")
    print(f"Attempts:   {expected_configs}")
    print()

    with tempfile.TemporaryDirectory(
        prefix="osu_adaptive_alpha_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        for replay_index, row in enumerate(
            cohort.itertuples(index=False),
            1,
        ):
            context = prepare_replay(
                row,
                args.dataset_dir,
                beatmaps,
                offsets,
            )
            performance_category = str(row.performance_category)
            capacity = nominal_capacity_bits(
                len(context.original_residuals),
                N_VALUE,
            )
            num_bits = message_length_for_fraction(
                capacity,
                PAYLOAD_FRACTION,
            )
            message = deterministic_message(
                replay_file=context.replay_file,
                beatmap_hash=context.beatmap_hash,
                alpha=0.0,
                n_value=N_VALUE,
                payload_fraction=PAYLOAD_FRACTION,
                num_bits=num_bits,
                seed=args.seed,
            )

            replay_results: list[dict] = []
            replay_features: list[dict] = []

            for method in METHODS:
                applied_alpha = alpha_for_method(
                    method,
                    performance_category,
                )
                config_id = experiment_config_id(
                    experiment_version=EXPERIMENT_VERSION,
                    policy_id=POLICY_ID,
                    policy_definition=POLICY_DEFINITION,
                    method=method,
                    replay_file=context.replay_file,
                    beatmap_hash=context.beatmap_hash,
                    performance_category=performance_category,
                    applied_alpha=applied_alpha,
                    n_frames_per_bit=N_VALUE,
                    payload_fraction=PAYLOAD_FRACTION,
                    message_bits=num_bits,
                    message_seed=args.seed,
                    pn_key_id=key_id(PN_KEY),
                    hit_margin_ms=args.hit_margin_ms,
                    offset_ms=context.offset_ms,
                    hit_window_ms=context.hit_window_ms,
                )
                if config_id in completed:
                    continue

                try:
                    result, clean, stego = run_one(
                        context=context,
                        message=message,
                        alpha=applied_alpha,
                        n_value=N_VALUE,
                        payload_fraction=PAYLOAD_FRACTION,
                        layout="prefix",
                        pn_key=PN_KEY,
                        layout_key=PREFIX_LAYOUT_KEY,
                        message_seed=args.seed,
                        config_id=config_id,
                        hit_margin_ms=args.hit_margin_ms,
                        temp_dir=temp_dir,
                    )
                    common = {
                        "experiment_version": EXPERIMENT_VERSION,
                        "method": method,
                        "policy_id": POLICY_ID,
                        "policy_definition": POLICY_DEFINITION,
                        "performance_category": performance_category,
                    }
                    result.update(common)
                    clean.update(common)
                    stego.update(common)
                    replay_results.append(result)
                    replay_features.extend([clean, stego])
                    new_ok += 1
                except Exception as exc:
                    replay_results.append(
                        {
                            "status": "ERROR",
                            "error": repr(exc),
                            "experiment_version": EXPERIMENT_VERSION,
                            "method": method,
                            "policy_id": POLICY_ID,
                            "policy_definition": POLICY_DEFINITION,
                            "performance_category": performance_category,
                            "config_id": config_id,
                            "message_seed": args.seed,
                            "pn_key": PN_KEY,
                            "pn_key_id": key_id(PN_KEY),
                            "layout_key": PREFIX_LAYOUT_KEY,
                            "layout_key_id": key_id(PREFIX_LAYOUT_KEY),
                            "hit_margin_ms": args.hit_margin_ms,
                            "layout": "prefix",
                            "category": context.category,
                            "replay_file": context.replay_file,
                            "player": context.player,
                            "beatmap_hash": context.beatmap_hash,
                            "alpha": applied_alpha,
                            "n_frames_per_bit": N_VALUE,
                            "payload_fraction": PAYLOAD_FRACTION,
                        }
                    )
                    print(
                        f"ERROR {context.replay_file[:12]} {method}: {exc!r}"
                    )

            replace_configs_rows(
                args.output,
                RESULT_FIELDS,
                replay_results,
            )
            replace_configs_rows(
                args.features,
                FEATURE_FIELDS,
                replay_features,
            )

            if replay_index % 25 == 0:
                print(
                    f"{replay_index}/{len(cohort)} replay-eva | new OK={new_ok}"
                )

    results = pd.read_csv(args.output)
    features = pd.read_csv(args.features)
    selected_ids = set(results[
        (results["experiment_version"] == EXPERIMENT_VERSION)
        & (results["message_seed"] == args.seed)
        & np.isclose(results["hit_margin_ms"], args.hit_margin_ms)
    ]["config_id"].astype(str))
    results = results[results["config_id"].astype(str).isin(selected_ids)].copy()
    features = features[
        features["config_id"].astype(str).isin(selected_ids)
    ].copy()
    validate_complete_rows(results, features, expected_configs)

    auc, paired = evaluate_methods(
        results,
        features,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    args.auc_output.parent.mkdir(parents=True, exist_ok=True)
    auc.to_csv(args.auc_output, index=False)
    paired.to_csv(args.paired_output, index=False)

    print("\nPAIRED ADAPTIVE RESULT")
    print(
        paired[
            [
                "control_method",
                "cv_group_column",
                "ber_control",
                "ber_adaptive",
                "delta_ber_adaptive_minus_control",
                "auc_control",
                "auc_adaptive",
                "delta_auc_adaptive_minus_control",
            ]
        ].to_string(index=False)
    )
    print("\nFajlovi:")
    for path in (
        args.output,
        args.features,
        args.auc_output,
        args.paired_output,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
