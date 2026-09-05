"""End-to-end Random Forest stegoanalysis za osu! timing residuale.

Za svaku izabranu (alpha, N) konfiguraciju:
1. za svaki clean replay računamo note-indeksirane residuale;
2. pravimo stvarni stego .osr ISTIM pipeline-om kao BER eksperiment;
3. reload + rematching;
4. iz clean i stego residuala izvlačimo samo planirane karakteristike:
      - variance
      - excess kurtosis
      - lag-1 autocorrelation
5. Random Forest evaluiramo 5-fold GROUP cross-validation-om.

Važna zaštita od leakage-a:
clean i stego verzija ISTOG replay-a imaju isti group id i nikada ne mogu
završiti u različitim train/test foldovima.

Default konfiguracije:
    alpha=10, N=16
    alpha=15, N=8
    alpha=15, N=16
    alpha=20, N=16

Glavna metrika:
    ROC-AUC, gde je 0.5 idealno za steganografsku neprimećenost.

Izlazi:
    steganalysis_features.csv
    steganalysis_auc_summary.csv
    steganalysis_auc_by_skill.csv
    steganalysis_oof_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path

from osu_stego.paths import (
    CONFIG_DIR,
    DATASET_DIR,
    METADATA_DIR,
    STEGANALYSIS_RESULTS_DIR,
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.stego.encoder import embed_message_indexed
from osu_stego.parsing.osr_writer import make_chronology_safe_shifts, write_stego_replay
from scripts.experiments.run_pilot_ber_sweep import (
    CATEGORIES,
    build_beatmap_index,
    deterministic_message,
    load_map_offsets,
    load_residuals,
    prepare_replay,
    stable_seed,
)


DEFAULT_CONFIGS = [
    (10.0, 16),
    (15.0, 8),
    (15.0, 16),
    (20.0, 16),
]

FEATURE_COLUMNS = [
    "variance",
    "excess_kurtosis",
    "autocorrelation_lag1",
]

FEATURE_FIELDS = [
    "category",
    "replay_file",
    "player",
    "beatmap_hash",
    "alpha",
    "n_frames_per_bit",
    "label",
    "variance",
    "excess_kurtosis",
    "autocorrelation_lag1",
    "n_valid_residuals",
    "num_notes",
    "active_carrier_fraction",
    "dropped_for_hit_window",
    "dropped_for_chronology",
    "delta_unmatched",
]


def parse_configs(values: list[str] | None) -> list[tuple[float, int]]:
    if not values:
        return DEFAULT_CONFIGS.copy()

    configs: list[tuple[float, int]] = []

    for value in values:
        try:
            alpha_text, n_text = value.split(":", maxsplit=1)
            alpha = float(alpha_text)
            n_value = int(n_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Neispravna konfiguracija '{value}'. Koristi format alpha:N, "
                "npr. 15:16."
            ) from exc

        if alpha <= 0 or n_value <= 0:
            raise ValueError("alpha i N moraju biti > 0.")

        configs.append((alpha, n_value))

    return configs


def residual_features(residuals: np.ndarray) -> dict[str, float | int]:
    """Tri planirane stegoanalysis karakteristike.

    Variance i kurtosis koriste sve validne residuale.

    Lag-1 autocorrelation čuva NOTE indeksiranje: u numerator ulaze samo
    susedne note za koje oba residuala postoje. Ne kompresujemo NaN pozicije,
    jer bi to lažno pravilo susede od nota koje zapravo nisu susedne.
    """
    values = np.asarray(residuals, dtype=np.float64)
    valid = ~np.isnan(values)
    clean_values = values[valid]

    if len(clean_values) < 4:
        raise ValueError(
            f"Premalo validnih residuala za statistiku: {len(clean_values)}."
        )

    mean = float(np.mean(clean_values))
    centered_valid = clean_values - mean

    variance = float(np.mean(centered_valid**2))

    if variance <= 0:
        raise ValueError("Varijansa residuala je 0; kurtosis nije definisan.")

    fourth_moment = float(np.mean(centered_valid**4))
    excess_kurtosis = fourth_moment / (variance**2) - 3.0

    centered_indexed = values - mean
    adjacent_valid = valid[:-1] & valid[1:]

    if not np.any(adjacent_valid):
        raise ValueError(
            "Nema nijednog para susednih validnih residuala za lag-1 autocorrelation."
        )

    numerator = float(
        np.sum(
            centered_indexed[:-1][adjacent_valid]
            * centered_indexed[1:][adjacent_valid]
        )
    )
    denominator = float(np.sum(centered_valid**2))

    if denominator <= 0:
        raise ValueError("Autocorrelation denominator je 0.")

    autocorrelation = numerator / denominator

    return {
        "variance": variance,
        "excess_kurtosis": float(excess_kurtosis),
        "autocorrelation_lag1": float(autocorrelation),
        "n_valid_residuals": int(len(clean_values)),
    }


def completed_pairs(
    feature_path: Path,
) -> set[tuple[str, float, int]]:
    """Pair je završen samo ako postoje i clean i stego red."""
    if not feature_path.exists():
        return set()

    frame = pd.read_csv(feature_path)

    required = {
        "replay_file",
        "alpha",
        "n_frames_per_bit",
        "label",
    }
    if not required.issubset(frame.columns):
        return set()

    completed: set[tuple[str, float, int]] = set()

    for keys, group in frame.groupby(
        ["replay_file", "alpha", "n_frames_per_bit"]
    ):
        labels = set(group["label"].astype(int).tolist())

        if labels == {0, 1}:
            replay_file, alpha, n_value = keys
            completed.add(
                (
                    str(replay_file),
                    float(alpha),
                    int(n_value),
                )
            )

    return completed


def append_feature_pair(
    path: Path,
    clean_row: dict,
    stego_row: dict,
) -> None:
    is_new = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FEATURE_FIELDS,
        )

        if is_new:
            writer.writeheader()

        writer.writerow(
            {
                field: clean_row.get(field, "")
                for field in FEATURE_FIELDS
            }
        )
        writer.writerow(
            {
                field: stego_row.get(field, "")
                for field in FEATURE_FIELDS
            }
        )


def build_stego_residuals(
    context,
    alpha: float,
    n_frames_per_bit: int,
    message: np.ndarray,
    key: str,
    hit_margin_ms: float,
    temp_dir: Path,
) -> tuple[np.ndarray, dict[str, float | int]]:
    original = context.original_residuals
    note_frames = context.note_frame_indices
    num_bits = len(message)
    required = num_bits * n_frames_per_bit

    if required > len(original):
        raise ValueError(
            f"Nedovoljan kapacitet: potrebno {required}, dostupno {len(original)}."
        )

    target_stego = embed_message_indexed(
        original,
        message,
        key,
        alpha,
        n_frames_per_bit,
        quantize=True,
    )

    requested_shifts = np.zeros(
        required,
        dtype=np.int64,
    )

    base_usable = (
        (note_frames[:required] != -1)
        & ~np.isnan(original[:required])
        & ~np.isnan(target_stego[:required])
    )

    requested_shifts[base_usable] = np.rint(
        target_stego[:required][base_usable]
        - original[:required][base_usable]
    ).astype(np.int64)

    originally_requested = (
        base_usable
        & (requested_shifts != 0)
    )

    limit = max(
        0.0,
        context.hit_window_ms - hit_margin_ms,
    )

    safe_for_hit_window = np.zeros(
        required,
        dtype=bool,
    )
    safe_for_hit_window[base_usable] = (
        np.abs(
            original[:required][base_usable]
            + requested_shifts[base_usable]
        )
        <= limit
    )

    dropped_hit_window = int(
        np.sum(
            originally_requested
            & ~safe_for_hit_window
        )
    )

    requested_shifts[
        originally_requested
        & ~safe_for_hit_window
    ] = 0

    target_positions = np.flatnonzero(
        base_usable
        & (requested_shifts != 0)
    )
    target_frames = note_frames[target_positions]
    requested_target_shifts = requested_shifts[target_positions]

    safe_target_shifts, dropped_chronology = (
        make_chronology_safe_shifts(
            context.replay,
            target_frames,
            requested_target_shifts,
        )
    )

    active_mask = safe_target_shifts != 0

    output_path = temp_dir / (
        f"{context.osr_path.stem}"
        f"_a{alpha:g}"
        f"_n{n_frames_per_bit}"
        "_steganalysis.osr"
    )

    write_stego_replay(
        str(context.osr_path),
        str(output_path),
        target_frames[active_mask],
        safe_target_shifts[active_mask],
    )

    stego_residuals, _, _ = load_residuals(
        output_path,
        context.beatmap,
        context.offset_ms,
        context.hit_window_ms,
    )

    original_unmatched = int(
        np.isnan(original).sum()
    )
    stego_unmatched = int(
        np.isnan(stego_residuals).sum()
    )

    requested_count = int(
        np.sum(originally_requested)
    )
    active_count = int(
        np.sum(active_mask)
    )

    diagnostics = {
        "active_carrier_fraction": (
            active_count / requested_count
            if requested_count
            else 0.0
        ),
        "dropped_for_hit_window": dropped_hit_window,
        "dropped_for_chronology": dropped_chronology,
        "delta_unmatched": (
            stego_unmatched - original_unmatched
        ),
    }

    return stego_residuals, diagnostics


def shuffled_group_ids(
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Reproducibilno permutuje group identitete pre GroupKFold-a.

    Ne razdvaja parove; samo sprečava da fold raspored zavisi od leksikografskog
    redosleda replay imena.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    mapping = {
        group: index
        for index, group in enumerate(shuffled.tolist())
    }

    return np.asarray(
        [mapping[group] for group in groups.tolist()],
        dtype=np.int64,
    )


def bootstrap_auc_by_group(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    unique_groups = np.unique(groups)

    row_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups.tolist()
    }

    rng = np.random.default_rng(seed)
    aucs: list[float] = []

    for _ in range(iterations):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        sampled_rows = np.concatenate(
            [
                row_indices[group]
                for group in sampled_groups.tolist()
            ]
        )

        sampled_labels = labels[sampled_rows]

        # Svaki pair ima clean+stego, pa bi obe klase trebalo uvek da postoje.
        if len(np.unique(sampled_labels)) < 2:
            continue

        aucs.append(
            float(
                roc_auc_score(
                    sampled_labels,
                    probabilities[sampled_rows],
                )
            )
        )

    if not aucs:
        return float("nan"), float("nan")

    lower, upper = np.percentile(
        np.asarray(aucs, dtype=np.float64),
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def evaluate_subset(
    frame: pd.DataFrame,
    scope: str,
    alpha: float | None,
    n_value: int,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
    cv_group_column: str = "replay_file",
    classifier_jobs: int = -1,
) -> tuple[dict, pd.DataFrame]:
    subset = frame[
        frame["n_frames_per_bit"] == n_value
    ].copy()
    if alpha is not None:
        subset = subset[
            subset["alpha"] == alpha
        ].copy()

    if scope != "ALL":
        subset = subset[
            subset["category"] == scope
        ].copy()

    if cv_group_column not in subset.columns:
        raise ValueError(
            f"Nedostaje CV group kolona: {cv_group_column}."
        )

    pair_labels = (
        subset.groupby("replay_file")["label"]
        .apply(lambda values: tuple(sorted(values.astype(int).tolist())))
    )
    valid_replays = pair_labels[
        pair_labels == (0, 1)
    ].index

    subset = subset[
        subset["replay_file"].isin(valid_replays)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"Nema validnih clean/stego parova za {scope}, alpha={alpha}, N={n_value}."
        )

    num_pairs = subset["replay_file"].nunique()
    num_cv_groups = subset[cv_group_column].nunique()
    n_splits = min(folds, num_cv_groups)

    if n_splits < 2:
        raise ValueError(
            f"Premalo {cv_group_column} grupa ({num_cv_groups}) "
            "za cross-validation."
        )

    X = subset[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = subset["label"].to_numpy(dtype=np.int8)
    replay_groups = subset["replay_file"].astype(str).to_numpy()
    raw_cv_groups = subset[cv_group_column].astype(str).to_numpy()
    fold_seed = (
        stable_seed(seed, scope, alpha, n_value, "folds")
        if cv_group_column == "replay_file"
        else stable_seed(
            seed,
            scope,
            alpha,
            n_value,
            cv_group_column,
            "folds",
        )
    )
    groups = shuffled_group_ids(
        raw_cv_groups,
        fold_seed,
    )

    if np.any(~np.isfinite(X)):
        raise ValueError(
            f"Feature matrica sadrži NaN/inf za {scope}, alpha={alpha}, N={n_value}."
        )

    splitter = GroupKFold(
        n_splits=n_splits,
    )

    probabilities = np.full(
        len(subset),
        np.nan,
        dtype=np.float64,
    )

    importances: list[np.ndarray] = []

    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups),
        1,
    ):
        rf_seed_parts: tuple[object, ...] = (
            seed,
            scope,
            alpha,
            n_value,
            fold_index,
            "rf",
        )
        if cv_group_column != "replay_file":
            rf_seed_parts = (
                seed,
                scope,
                alpha,
                n_value,
                cv_group_column,
                fold_index,
                "rf",
            )

        classifier = RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=stable_seed(*rf_seed_parts),
            n_jobs=classifier_jobs,
        )

        classifier.fit(
            X[train_index],
            y[train_index],
        )

        probabilities[test_index] = classifier.predict_proba(
            X[test_index]
        )[:, 1]

        importances.append(
            classifier.feature_importances_.astype(
                np.float64
            )
        )

    if np.any(np.isnan(probabilities)):
        raise RuntimeError(
            "Nisu svi redovi dobili out-of-fold verovatnoću."
        )

    auc = float(
        roc_auc_score(
            y,
            probabilities,
        )
    )

    bootstrap_seed = (
        stable_seed(seed, scope, alpha, n_value, "bootstrap")
        if cv_group_column == "replay_file"
        else stable_seed(
            seed,
            scope,
            alpha,
            n_value,
            cv_group_column,
            "bootstrap",
        )
    )
    ci_low, ci_high = bootstrap_auc_by_group(
        y,
        probabilities,
        replay_groups,
        bootstrap_seed,
        bootstrap_iterations,
    )

    mean_importance = np.mean(
        np.vstack(importances),
        axis=0,
    )

    clean = subset[
        subset["label"] == 0
    ]
    stego = subset[
        subset["label"] == 1
    ]

    summary = {
        "scope": scope,
        "alpha": float(alpha) if alpha is not None else float("nan"),
        "n_frames_per_bit": int(n_value),
        "replay_pairs": int(num_pairs),
        "cv_group_column": cv_group_column,
        "cv_groups": int(num_cv_groups),
        "rows": int(len(subset)),
        "folds": int(n_splits),
        "roc_auc": auc,
        "auc_ci95_low": ci_low,
        "auc_ci95_high": ci_high,
        "auc_distance_from_0_5": abs(auc - 0.5),
        "importance_variance": float(mean_importance[0]),
        "importance_excess_kurtosis": float(mean_importance[1]),
        "importance_autocorrelation_lag1": float(mean_importance[2]),
        "clean_mean_variance": float(clean["variance"].mean()),
        "stego_mean_variance": float(stego["variance"].mean()),
        "clean_mean_excess_kurtosis": float(
            clean["excess_kurtosis"].mean()
        ),
        "stego_mean_excess_kurtosis": float(
            stego["excess_kurtosis"].mean()
        ),
        "clean_mean_autocorrelation_lag1": float(
            clean["autocorrelation_lag1"].mean()
        ),
        "stego_mean_autocorrelation_lag1": float(
            stego["autocorrelation_lag1"].mean()
        ),
    }

    predictions = subset[
        [
            "category",
            "replay_file",
            "beatmap_hash",
            "alpha",
            "n_frames_per_bit",
            "label",
        ]
    ].copy()
    predictions["scope"] = scope
    predictions["oof_probability_stego"] = probabilities

    return summary, predictions


def generate_features(
    results: pd.DataFrame,
    dataset_dir: Path,
    map_offsets_path: Path,
    feature_path: Path,
    configs: list[tuple[float, int]],
    bits: int,
    key: str,
    seed: int,
    hit_margin_ms: float,
    overwrite: bool,
    limit_per_category: int | None,
) -> None:
    if overwrite and feature_path.exists():
        feature_path.unlink()

    completed = completed_pairs(
        feature_path
    )

    offsets = load_map_offsets(
        map_offsets_path
    )
    beatmaps = build_beatmap_index(
        dataset_dir
    )

    working = results.sort_values(
        ["skill_category", "replay_file"]
    ).copy()

    if limit_per_category is not None:
        parts = []
        for category in CATEGORIES:
            parts.append(
                working[
                    working["skill_category"] == category
                ].head(limit_per_category)
            )
        working = pd.concat(
            parts,
            ignore_index=True,
        )

    total_pairs = len(working) * len(configs)
    done_now = 0
    errors = 0
    start = time.perf_counter()

    print("=" * 100)
    print("GENERISANJE CLEAN/STego FEATURE PAROVA")
    print("=" * 100)
    print(f"Replay-eva:     {len(working)}")
    print(f"Konfiguracije:  {configs}")
    print(f"Parova ukupno:  {total_pairs}")
    if completed:
        print(f"Već završeno:   {len(completed)}")
    print()

    with tempfile.TemporaryDirectory(
        prefix="osu_steganalysis_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        for replay_index, row in enumerate(
            working.itertuples(index=False),
            1,
        ):
            try:
                context = prepare_replay(
                    row,
                    dataset_dir,
                    beatmaps,
                    offsets,
                )
            except Exception as exc:
                errors += len(configs)
                print(
                    f"[{replay_index:03d}/{len(working):03d}] "
                    f"BASELINE ERROR {row.replay_file}: {exc!r}"
                )
                continue

            clean_stats = residual_features(
                context.original_residuals
            )

            message = deterministic_message(
                context.beatmap_hash,
                context.replay_file,
                seed,
                bits,
            )

            for alpha, n_value in configs:
                pair_key = (
                    context.replay_file,
                    float(alpha),
                    int(n_value),
                )

                if pair_key in completed:
                    continue

                try:
                    stego_residuals, diagnostics = (
                        build_stego_residuals(
                            context=context,
                            alpha=alpha,
                            n_frames_per_bit=n_value,
                            message=message,
                            key=key,
                            hit_margin_ms=hit_margin_ms,
                            temp_dir=temp_dir,
                        )
                    )

                    stego_stats = residual_features(
                        stego_residuals
                    )

                    common = {
                        "category": context.category,
                        "replay_file": context.replay_file,
                        "player": context.player,
                        "beatmap_hash": context.beatmap_hash,
                        "alpha": float(alpha),
                        "n_frames_per_bit": int(n_value),
                        "num_notes": int(
                            len(context.original_residuals)
                        ),
                        "active_carrier_fraction": diagnostics[
                            "active_carrier_fraction"
                        ],
                        "dropped_for_hit_window": diagnostics[
                            "dropped_for_hit_window"
                        ],
                        "dropped_for_chronology": diagnostics[
                            "dropped_for_chronology"
                        ],
                        "delta_unmatched": diagnostics[
                            "delta_unmatched"
                        ],
                    }

                    clean_row = {
                        **common,
                        **clean_stats,
                        "label": 0,
                    }
                    stego_row = {
                        **common,
                        **stego_stats,
                        "label": 1,
                    }

                    append_feature_pair(
                        feature_path,
                        clean_row,
                        stego_row,
                    )
                    done_now += 1

                except Exception as exc:
                    errors += 1
                    print(
                        f"  ERROR {context.replay_file[:12]} "
                        f"a={alpha:g} N={n_value}: {exc!r}"
                    )

            if (
                replay_index % 25 == 0
                or replay_index == len(working)
            ):
                elapsed = time.perf_counter() - start
                print(
                    f"[{replay_index:03d}/{len(working):03d}] "
                    f"novi parovi={done_now} errors={errors} "
                    f"elapsed={elapsed / 60:.1f} min"
                )

    print()
    print(f"Feature CSV: {feature_path}")


def evaluate_all(
    feature_path: Path,
    configs: list[tuple[float, int]],
    overall_path: Path,
    skill_path: Path,
    predictions_path: Path,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> None:
    frame = pd.read_csv(
        feature_path
    )

    summaries: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []

    scopes = ["ALL", *CATEGORIES]

    for alpha, n_value in configs:
        for scope in scopes:
            summary, predictions = evaluate_subset(
                frame=frame,
                scope=scope,
                alpha=alpha,
                n_value=n_value,
                folds=folds,
                trees=trees,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
            )

            summaries.append(summary)
            prediction_parts.append(predictions)

            print(
                f"{scope:10s} | "
                f"a={alpha:4.0f} N={n_value:2d} | "
                f"AUC={summary['roc_auc']:.4f} "
                f"[{summary['auc_ci95_low']:.4f}, "
                f"{summary['auc_ci95_high']:.4f}]"
            )

    summary_frame = pd.DataFrame(
        summaries
    )

    overall = summary_frame[
        summary_frame["scope"] == "ALL"
    ].copy()
    by_skill = summary_frame[
        summary_frame["scope"] != "ALL"
    ].copy()

    overall.to_csv(
        overall_path,
        index=False,
    )
    by_skill.to_csv(
        skill_path,
        index=False,
    )

    pd.concat(
        prediction_parts,
        ignore_index=True,
    ).to_csv(
        predictions_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Random Forest stegoanalysis: clean vs stvarni round-trip stego residuali."
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
        "--features",
        type=Path,
        default=STEGANALYSIS_RESULTS_DIR / "steganalysis_features.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=STEGANALYSIS_RESULTS_DIR / "steganalysis_auc_summary.csv",
    )
    parser.add_argument(
        "--skill-summary",
        type=Path,
        default=STEGANALYSIS_RESULTS_DIR / "steganalysis_auc_by_skill.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=STEGANALYSIS_RESULTS_DIR / "steganalysis_oof_predictions.csv",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=None,
        help=(
            "Konfiguracije u formatu alpha:N, npr. "
            "--config 10:16 15:8 15:16 20:16"
        ),
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
        help="Isti default ključ kao BER eksperiment.",
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
        "--overwrite-features",
        action="store_true",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="Samo za smoke test; default koristi ceo dataset.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Preskoči generisanje .osr i evaluiraj postojeći feature CSV.",
    )

    args = parser.parse_args()

    configs = parse_configs(
        args.config
    )

    if not args.results.is_file():
        raise FileNotFoundError(
            f"Ne postoji {args.results}"
        )

    if not args.evaluate_only:
        results = pd.read_csv(
            args.results
        )

        generate_features(
            results=results,
            dataset_dir=args.dataset_dir,
            map_offsets_path=args.map_offsets,
            feature_path=args.features,
            configs=configs,
            bits=args.bits,
            key=args.key,
            seed=args.seed,
            hit_margin_ms=args.hit_margin_ms,
            overwrite=args.overwrite_features,
            limit_per_category=args.limit_per_category,
        )

    if not args.features.is_file():
        raise FileNotFoundError(
            f"Ne postoji feature CSV: {args.features}"
        )

    print("\n" + "=" * 100)
    print("RANDOM FOREST GROUP-CV STEGOANALYSIS")
    print("=" * 100)

    evaluate_all(
        feature_path=args.features,
        configs=configs,
        overall_path=args.summary,
        skill_path=args.skill_summary,
        predictions_path=args.predictions,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )

    print("\n" + "=" * 100)
    print("GOTOVO")
    print("=" * 100)
    print(f"Features:      {args.features}")
    print(f"Overall AUC:   {args.summary}")
    print(f"AUC po skillu:{args.skill_summary}")
    print(f"OOF predikcije:{args.predictions}")


if __name__ == "__main__":
    main()
