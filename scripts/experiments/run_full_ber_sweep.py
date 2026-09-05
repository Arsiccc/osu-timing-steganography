"""Full BER sweep preko celog čistog osu! dataset-a.

Namerno REUSE-uje iste funkcije iz run_pilot_ber_sweep.py koje su već prošle
pilot od 1200 eksperimenata. Time ne uvodimo novi eksperimentalni pipeline.

Default:
- svih 949 replay-eva iz results_v3_clean.csv
- alpha = [2, 5, 10, 15, 20] ms
- N = [8, 16, 32]
- poruka = 8 bita
- 14,235 pokušaja ukupno

Ako replay nema dovoljno note-pozicija za 8*N, konfiguracija se beleži kao
SKIP_CAPACITY umesto da se menja broj bitova ili da se replay tiho izbaci.

Run je resumable: ako full_ber_results.csv već postoji, OK i SKIP_CAPACITY
redovi se preskaču; ERROR redovi se pokušavaju ponovo.
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
    FULL_BER_RESULTS_DIR,
    METADATA_DIR,
)

import pandas as pd

from scripts.experiments.run_pilot_ber_sweep import (
    CATEGORIES,
    DEFAULT_ALPHAS,
    DEFAULT_N_VALUES,
    RESULT_FIELDS,
    append_result,
    build_beatmap_index,
    build_error_row,
    deterministic_message,
    load_map_offsets,
    prepare_replay,
    print_summary,
    run_experiment,
    write_summaries,
)


def completed_configurations(
    output_path: Path,
) -> set[tuple[str, float, int]]:
    """OK i SKIP_CAPACITY su završeni; ERROR se retry-uje pri resume-u."""
    if not output_path.exists():
        return set()

    completed: set[tuple[str, float, int]] = set()

    with output_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            status = row.get("status", "")

            if status not in {"OK", "SKIP_CAPACITY"}:
                continue

            try:
                completed.add(
                    (
                        row["replay_file"],
                        float(row["alpha"]),
                        int(row["n_frames_per_bit"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

    return completed


def capacity_skip_row(
    context,
    alpha: float,
    n_frames_per_bit: int,
    message_bits: int,
) -> dict:
    required_positions = message_bits * n_frames_per_bit
    available_positions = len(context.original_residuals)

    return {
        "status": "SKIP_CAPACITY",
        "error": (
            f"Potrebno {required_positions} note-pozicija, "
            f"dostupno {available_positions}."
        ),
        "category": context.category,
        "replay_file": context.replay_file,
        "player": context.player,
        "beatmap_hash": context.beatmap_hash,
        "alpha": float(alpha),
        "n_frames_per_bit": int(n_frames_per_bit),
        "message_bits": int(message_bits),
        "num_notes": available_positions,
        "runtime_seconds": 0.0,
    }


def write_status_summary(
    output_path: Path,
    status_summary_path: Path,
) -> None:
    frame = pd.read_csv(output_path)

    if frame.empty:
        return

    summary = (
        frame.groupby(
            [
                "alpha",
                "n_frames_per_bit",
                "status",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            [
                "n_frames_per_bit",
                "alpha",
                "status",
            ]
        )
    )

    summary.to_csv(
        status_summary_path,
        index=False,
    )


def validate_full_dataset(
    results: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "skill_category",
        "replay_file",
        "beatmap_hash",
        "num_notes",
    }

    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(
            f"results CSV nema kolone: {sorted(missing)}"
        )

    duplicate_files = results[
        results["replay_file"].duplicated(keep=False)
    ]

    if not duplicate_files.empty:
        raise ValueError(
            "results_v3_clean.csv sadrži duplikate replay_file vrednosti."
        )

    # Deterministički redosled olakšava resume i pregled log-a.
    return results.sort_values(
        ["skill_category", "replay_file"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full alpha x N BER sweep nad celim čistim dataset-om "
            "sa pravim .osr round-tripom."
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
        "--output",
        type=Path,
        default=FULL_BER_RESULTS_DIR / "full_ber_results.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=FULL_BER_RESULTS_DIR / "full_ber_summary.csv",
    )
    parser.add_argument(
        "--skill-summary",
        type=Path,
        default=FULL_BER_RESULTS_DIR / "full_ber_summary_by_skill.csv",
    )
    parser.add_argument(
        "--status-summary",
        type=Path,
        default=FULL_BER_RESULTS_DIR / "full_ber_status_summary.csv",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=DEFAULT_N_VALUES,
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hit-margin-ms",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--key",
        type=str,
        default="pilot-ber-key-v1",
        help=(
            "Namerno isti default key kao pilot, da se menja samo veličina uzorka."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Namerno isti seed kao pilot; poruka za isti replay ostaje ista."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Obriši postojeći full output i kreni od početka.",
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

    if not args.map_offsets.is_file():
        raise FileNotFoundError(
            f"Ne postoji offset JSON: {args.map_offsets}"
        )

    if args.bits <= 0:
        raise ValueError("--bits mora biti > 0.")

    if not args.n_values or min(args.n_values) <= 0:
        raise ValueError("Sve N vrednosti moraju biti > 0.")

    if not args.alphas or min(args.alphas) <= 0:
        raise ValueError("Sve alpha vrednosti moraju biti > 0.")

    if args.overwrite:
        for path in (
            args.output,
            args.summary,
            args.skill_summary,
            args.status_summary,
        ):
            if path.exists():
                path.unlink()

    results = validate_full_dataset(
        pd.read_csv(args.results)
    )

    offsets = load_map_offsets(args.map_offsets)
    beatmaps = build_beatmap_index(args.dataset_dir)

    total_replays = len(results)
    total_configurations = (
        len(args.alphas)
        * len(args.n_values)
    )
    total_attempts = (
        total_replays
        * total_configurations
    )

    capacity_counts: dict[int, int] = {}
    for n_value in args.n_values:
        required = args.bits * int(n_value)
        capacity_counts[int(n_value)] = int(
            (results["num_notes"] >= required).sum()
        )

    completed = completed_configurations(
        args.output
    )

    print("=" * 100)
    print("FULL BER SWEEP")
    print("=" * 100)
    print(f"Replay-eva:       {total_replays}")
    print(f"Alpha:            {args.alphas}")
    print(f"N:                {args.n_values}")
    print(f"Poruka:           {args.bits} bita")
    print(f"Ukupno pokušaja:  {total_attempts}")
    print()

    for n_value in args.n_values:
        enough = capacity_counts[int(n_value)]
        missing = total_replays - enough
        print(
            f"N={int(n_value):2d}: "
            f"{enough}/{total_replays} replay-eva ima kapacitet "
            f"({missing} SKIP po alpha vrednosti)"
        )

    if completed:
        print(f"\nVeć završeno/resume: {len(completed)} konfiguracija")

    print()

    attempt_counter = 0
    new_ok = 0
    new_skipped = 0
    new_errors = 0
    global_start = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="osu_full_ber_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        for replay_number, row in enumerate(
            results.itertuples(index=False),
            1,
        ):
            context = None
            baseline_error = None

            try:
                context = prepare_replay(
                    row,
                    args.dataset_dir,
                    beatmaps,
                    offsets,
                )
            except Exception as exc:
                baseline_error = exc

            message = deterministic_message(
                str(row.beatmap_hash),
                str(row.replay_file),
                args.seed,
                args.bits,
            )

            # Jedan kratak red po replay-u, ne po eksperimentu.
            print(
                f"[{replay_number:03d}/{total_replays:03d}] "
                f"{str(row.skill_category):10s} | "
                f"{str(row.replay_file)[:12]}"
            )

            for n_value in args.n_values:
                for alpha in args.alphas:
                    attempt_counter += 1

                    config_key = (
                        str(row.replay_file),
                        float(alpha),
                        int(n_value),
                    )

                    if config_key in completed:
                        continue

                    experiment_start = time.perf_counter()

                    if context is None:
                        result = build_error_row(
                            context=None,
                            row=row,
                            alpha=float(alpha),
                            n_frames_per_bit=int(n_value),
                            message_bits=args.bits,
                            error=RuntimeError(
                                f"Baseline priprema nije uspela: "
                                f"{baseline_error!r}"
                            ),
                            runtime_seconds=(
                                time.perf_counter()
                                - experiment_start
                            ),
                        )
                        new_errors += 1

                    elif (
                        args.bits * int(n_value)
                        > len(context.original_residuals)
                    ):
                        result = capacity_skip_row(
                            context=context,
                            alpha=float(alpha),
                            n_frames_per_bit=int(n_value),
                            message_bits=args.bits,
                        )
                        new_skipped += 1

                    else:
                        try:
                            result = run_experiment(
                                context=context,
                                alpha=float(alpha),
                                n_frames_per_bit=int(n_value),
                                message=message,
                                key=args.key,
                                hit_margin_ms=args.hit_margin_ms,
                                temp_dir=temp_dir,
                            )
                            new_ok += 1
                        except Exception as exc:
                            result = build_error_row(
                                context=context,
                                row=row,
                                alpha=float(alpha),
                                n_frames_per_bit=int(n_value),
                                message_bits=args.bits,
                                error=exc,
                                runtime_seconds=(
                                    time.perf_counter()
                                    - experiment_start
                                ),
                            )
                            new_errors += 1

                    append_result(
                        args.output,
                        result,
                    )

                    if (
                        attempt_counter % 100 == 0
                        or result["status"] == "ERROR"
                    ):
                        elapsed = (
                            time.perf_counter()
                            - global_start
                        )
                        print(
                            f"  progress "
                            f"{attempt_counter:5d}/{total_attempts} | "
                            f"new OK={new_ok} "
                            f"SKIP={new_skipped} "
                            f"ERR={new_errors} | "
                            f"elapsed={elapsed / 60:.1f} min"
                        )

    write_summaries(
        args.output,
        args.summary,
        args.skill_summary,
    )
    write_status_summary(
        args.output,
        args.status_summary,
    )

    print_summary(args.output)

    elapsed = time.perf_counter() - global_start

    frame = pd.read_csv(args.output)
    status_counts = (
        frame["status"]
        .value_counts()
        .to_dict()
    )

    print("\n" + "=" * 100)
    print("FULL RUN GOTOV")
    print("=" * 100)
    print(f"Ukupno vreme ovog pokretanja: {elapsed / 60:.1f} min")
    print(f"Statusi u CSV-u:              {status_counts}")
    print(f"Detaljni rezultati:           {args.output}")
    print(f"Overall summary:              {args.summary}")
    print(f"Skill summary:                {args.skill_summary}")
    print(f"Status summary:               {args.status_summary}")


if __name__ == "__main__":
    main()
