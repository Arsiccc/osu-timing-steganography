"""Grupisanje osu! replay-eva po kvalitetu KONKRETNOG odigranog replay-a.

Kategorije se ne zasnivaju na ranku igrača, već na osu!standard accuracy-ju
unutar ISTE beatmape.

Zašto unutar iste beatmape?
Globalnih 99% na jednoj mapi i 95% na drugoj nisu fer direktno poređenje.
Percentile unutar beatmap_hash-a uklanja veliki deo tog map-level confounda.

Accuracy za osu!standard:
    (300*n300 + 100*n100 + 50*n50)
    --------------------------------
    300*(n300+n100+n50+nmiss)

Izlazi:
    performance_groups.csv
    performance_group_summary.csv
    performance_group_by_old_skill.csv
    performance_group_by_map.csv

Napomena:
- Ne kopira .osr fajlove; pravi "logički dataset" preko metadata CSV-a.
- Grupe se mogu kasnije koristiti za BER i stegoanalysis bez dupliranja fajlova.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from osu_stego.paths import (
    DATASET_DIR,
    METADATA_DIR,
    PERFORMANCE_RESULTS_DIR,
)

import numpy as np
import osrparse
import pandas as pd


PERFORMANCE_ORDER = [
    "Very Poor",
    "Poor",
    "Good",
    "Very Good",
]


def osu_standard_accuracy(
    count_300: int,
    count_100: int,
    count_50: int,
    count_miss: int,
) -> float:
    total = (
        int(count_300)
        + int(count_100)
        + int(count_50)
        + int(count_miss)
    )

    if total <= 0:
        return float("nan")

    weighted = (
        300 * int(count_300)
        + 100 * int(count_100)
        + 50 * int(count_50)
    )

    return weighted / (300.0 * total)


def mods_value(replay: osrparse.Replay) -> int:
    mods = replay.mods

    if hasattr(mods, "value"):
        return int(mods.value)

    return int(mods)


def find_replay_path(
    dataset_dir: Path,
    old_skill_category: str,
    replay_file: str,
) -> Path:
    direct = (
        dataset_dir
        / old_skill_category
        / "osr"
        / replay_file
    )

    if direct.is_file():
        return direct

    matches = list(
        dataset_dir.glob(
            f"*/osr/{replay_file}"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Nije pronađen replay: {replay_file}"
        )

    raise RuntimeError(
        f"Replay {replay_file} postoji na više mesta: {matches}"
    )


def load_replay_metadata(
    replay_path: Path,
) -> dict:
    replay = osrparse.Replay.from_path(
        replay_path
    )

    count_300 = int(replay.count_300)
    count_100 = int(replay.count_100)
    count_50 = int(replay.count_50)
    count_miss = int(replay.count_miss)

    total_judgements = (
        count_300
        + count_100
        + count_50
        + count_miss
    )

    accuracy = osu_standard_accuracy(
        count_300=count_300,
        count_100=count_100,
        count_50=count_50,
        count_miss=count_miss,
    )

    miss_rate = (
        count_miss / total_judgements
        if total_judgements > 0
        else float("nan")
    )

    return {
        "player": str(replay.username),
        "count_300": count_300,
        "count_100": count_100,
        "count_50": count_50,
        "count_miss": count_miss,
        "total_judgements": total_judgements,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "miss_rate": miss_rate,
        "score": int(replay.score),
        "max_combo": int(replay.max_combo),
        "mods_value": mods_value(replay),
    }


def performance_percentile(
    accuracies: pd.Series,
) -> pd.Series:
    """0 = najlošiji replay na mapi, 100 = najbolji.

    Ties dobijaju prosečan rank, dakle replay-evi sa identičnim accuracy-jem
    ne razdvajaju se proizvoljno.
    """
    count = len(accuracies)

    if count <= 1:
        return pd.Series(
            np.full(count, 50.0),
            index=accuracies.index,
            dtype=float,
        )

    ranks = accuracies.rank(
        method="average",
        ascending=True,
    )

    return (
        (ranks - 1.0)
        / (count - 1.0)
        * 100.0
    )


def category_from_percentile(
    percentile: float,
) -> str:
    if percentile < 25.0:
        return "Very Poor"

    if percentile < 50.0:
        return "Poor"

    if percentile < 75.0:
        return "Good"

    return "Very Good"


def add_performance_groups(
    frame: pd.DataFrame,
    min_map_replays: int,
) -> pd.DataFrame:
    result = frame.copy()

    map_sizes = result.groupby(
        "beatmap_hash"
    )["replay_file"].transform("size")

    result["replays_on_same_map"] = (
        map_sizes.astype(int)
    )

    result["performance_percentile"] = np.nan
    result["performance_category"] = (
        "INSUFFICIENT_MAP_REPLAYS"
    )

    eligible = (
        result["replays_on_same_map"]
        >= min_map_replays
    )

    for _, indices in (
        result[eligible]
        .groupby("beatmap_hash")
        .groups.items()
    ):
        index_list = list(indices)

        percentiles = performance_percentile(
            result.loc[
                index_list,
                "accuracy",
            ]
        )

        result.loc[
            index_list,
            "performance_percentile",
        ] = percentiles

        result.loc[
            index_list,
            "performance_category",
        ] = percentiles.map(
            category_from_percentile
        )

    return result


def write_summaries(
    frame: pd.DataFrame,
    summary_path: Path,
    skill_path: Path,
    map_path: Path,
) -> None:
    valid = frame[
        frame["performance_category"]
        .isin(PERFORMANCE_ORDER)
    ].copy()

    valid["performance_category"] = pd.Categorical(
        valid["performance_category"],
        categories=PERFORMANCE_ORDER,
        ordered=True,
    )

    summary = (
        valid.groupby(
            "performance_category",
            observed=True,
        )
        .agg(
            replay_count=(
                "replay_file",
                "size",
            ),
            beatmap_count=(
                "beatmap_hash",
                "nunique",
            ),
            mean_accuracy_percent=(
                "accuracy_percent",
                "mean",
            ),
            median_accuracy_percent=(
                "accuracy_percent",
                "median",
            ),
            mean_miss_rate=(
                "miss_rate",
                "mean",
            ),
            median_miss_count=(
                "count_miss",
                "median",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    by_skill = (
        valid.groupby(
            [
                "old_skill_category",
                "performance_category",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="replay_count")
    )

    totals = (
        by_skill.groupby(
            "old_skill_category"
        )["replay_count"]
        .transform("sum")
    )

    by_skill["fraction_within_old_skill"] = (
        by_skill["replay_count"]
        / totals
    )

    by_skill.to_csv(
        skill_path,
        index=False,
    )

    by_map = (
        valid.groupby(
            [
                "beatmap_hash",
                "performance_category",
            ],
            observed=True,
        )
        .agg(
            replay_count=(
                "replay_file",
                "size",
            ),
            min_accuracy_percent=(
                "accuracy_percent",
                "min",
            ),
            max_accuracy_percent=(
                "accuracy_percent",
                "max",
            ),
            mean_accuracy_percent=(
                "accuracy_percent",
                "mean",
            ),
        )
        .reset_index()
    )

    by_map.to_csv(
        map_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pravi performance-based logical dataset "
            "iz čistog osu! replay dataseta."
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
        help=(
            "Jedan red po aktivnom clean replay-u. "
            "Očekuje replay_file, beatmap_hash i skill/category kolonu."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METADATA_DIR / "performance_groups.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR / "performance_group_summary.csv",
    )
    parser.add_argument(
        "--skill-summary",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR / "performance_group_by_old_skill.csv",
    )
    parser.add_argument(
        "--map-summary",
        type=Path,
        default=PERFORMANCE_RESULTS_DIR / "performance_group_by_map.csv",
    )
    parser.add_argument(
        "--min-map-replays",
        type=int,
        default=5,
        help=(
            "Minimum replay-eva iste beatmape potreban "
            "za percentile grupisanje."
        ),
    )

    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Ne postoji dataset: {args.dataset_dir}"
        )

    if not args.results.is_file():
        raise FileNotFoundError(
            f"Ne postoji results CSV: {args.results}"
        )

    if args.min_map_replays < 2:
        raise ValueError(
            "--min-map-replays mora biti >= 2."
        )

    source = pd.read_csv(
        args.results
    )

    if "skill_category" in source.columns:
        old_category_column = "skill_category"
    elif "category" in source.columns:
        old_category_column = "category"
    else:
        raise ValueError(
            "Results CSV mora imati 'skill_category' "
            "ili 'category' kolonu."
        )

    required_columns = {
        "replay_file",
        "beatmap_hash",
        old_category_column,
    }

    missing = (
        required_columns
        - set(source.columns)
    )

    if missing:
        raise ValueError(
            f"Nedostaju kolone: {sorted(missing)}"
        )

    source = (
        source[
            [
                old_category_column,
                "replay_file",
                "beatmap_hash",
            ]
        ]
        .drop_duplicates(
            subset=["replay_file"]
        )
        .rename(
            columns={
                old_category_column:
                    "old_skill_category",
            }
        )
        .reset_index(drop=True)
    )

    print("=" * 100)
    print("BUILD PERFORMANCE GROUPS")
    print("=" * 100)
    print(f"Replay-eva: {len(source)}")
    print(
        f"Beatmapa:   "
        f"{source['beatmap_hash'].nunique()}"
    )
    print()

    rows: list[dict] = []

    for index, row in enumerate(
        source.itertuples(index=False),
        1,
    ):
        replay_path = find_replay_path(
            dataset_dir=args.dataset_dir,
            old_skill_category=(
                row.old_skill_category
            ),
            replay_file=row.replay_file,
        )

        metadata = load_replay_metadata(
            replay_path
        )

        rows.append(
            {
                "replay_file": row.replay_file,
                "beatmap_hash": row.beatmap_hash,
                "old_skill_category": (
                    row.old_skill_category
                ),
                **metadata,
            }
        )

        if (
            index % 100 == 0
            or index == len(source)
        ):
            print(
                f"Učitano: {index}/{len(source)}"
            )

    frame = pd.DataFrame(
        rows
    )

    invalid_accuracy = frame[
        ~np.isfinite(frame["accuracy"])
    ]

    if not invalid_accuracy.empty:
        raise ValueError(
            f"{len(invalid_accuracy)} replay-eva "
            "nema validan accuracy."
        )

    frame = add_performance_groups(
        frame=frame,
        min_map_replays=(
            args.min_map_replays
        ),
    )

    category_order = {
        name: index
        for index, name in enumerate(
            PERFORMANCE_ORDER
        )
    }

    frame["_category_order"] = (
        frame["performance_category"]
        .map(category_order)
        .fillna(999)
    )

    frame = (
        frame.sort_values(
            [
                "_category_order",
                "beatmap_hash",
                "performance_percentile",
                "accuracy",
            ]
        )
        .drop(
            columns=["_category_order"]
        )
        .reset_index(drop=True)
    )

    frame.to_csv(
        args.output,
        index=False,
    )

    write_summaries(
        frame=frame,
        summary_path=args.summary,
        skill_path=args.skill_summary,
        map_path=args.map_summary,
    )

    valid = frame[
        frame["performance_category"]
        .isin(PERFORMANCE_ORDER)
    ]

    excluded = len(frame) - len(valid)

    print("\n" + "=" * 100)
    print("PERFORMANCE GROUP SUMMARY")
    print("=" * 100)

    counts = (
        valid["performance_category"]
        .value_counts()
        .reindex(
            PERFORMANCE_ORDER,
            fill_value=0,
        )
    )

    for category in PERFORMANCE_ORDER:
        group = valid[
            valid["performance_category"]
            == category
        ]

        print(
            f"{category:10s}: "
            f"{int(counts[category]):4d} replay-eva | "
            f"mean acc="
            f"{group['accuracy_percent'].mean():6.2f}%"
        )

    print(
        f"\nNedovoljno replay-eva na mapi: "
        f"{excluded}"
    )

    print("\nFajlovi:")
    print(f"  {args.output}")
    print(f"  {args.summary}")
    print(f"  {args.skill_summary}")
    print(f"  {args.map_summary}")


if __name__ == "__main__":
    main()
