"""Exact non-physical payload coverage audit for the frozen N=8 channel."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import METADATA_DIR, RESULTS_DIR
from osu_stego.stego.ecc import encode_hamming_8_4_secded
from scripts.experiments.adaptive_layout_strong_common import DEFAULT_PARTITION
from scripts.experiments.run_payload_sweep import (
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "capacity-coverage-v1"
N_VALUE = 8
BASE_FRACTION = 0.075
CAPS = (0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.50, 1.00)
CODE_LENGTH = 8
INFO_LENGTH = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minimum_fraction(capacity: int, target_bits: int) -> float:
    """Smallest representable fraction found under the production allocator."""
    if capacity < target_bits:
        return float("nan")
    candidate = float(target_bits / capacity)
    while message_length_for_fraction(capacity, candidate) < target_bits:
        candidate = float(np.nextafter(candidate, np.inf))
    while True:
        previous = float(np.nextafter(candidate, -np.inf))
        if previous <= 0 or message_length_for_fraction(capacity, previous) < target_bits:
            break
        candidate = previous
    return candidate


def shortened_code_rows() -> pd.DataFrame:
    """Exhaustively verify systematic shortenings of the frozen [8,4,4] code."""
    data_positions = (2, 4, 5, 6)
    rows: list[dict[str, object]] = []
    for shortened_by in range(4):
        fixed_info = tuple(range(shortened_by))
        removed_positions = tuple(data_positions[index] for index in fixed_info)
        codewords: list[np.ndarray] = []
        for free in itertools.product((-1, 1), repeat=INFO_LENGTH - shortened_by):
            info = np.full(INFO_LENGTH, -1, dtype=np.int8)
            info[shortened_by:] = np.asarray(free, dtype=np.int8)
            full = encode_hamming_8_4_secded(info)
            codewords.append(np.delete(full, removed_positions))
        words = np.asarray(codewords, dtype=np.int8)
        distances = [
            int(np.sum(words[left] != words[right]))
            for left in range(len(words))
            for right in range(left + 1, len(words))
        ]
        minimum_distance = min(distances) if distances else len(words[0])
        valid_single = 0
        valid_double_detect = 0
        total_single = len(words) * words.shape[1]
        total_double = len(words) * (words.shape[1] * (words.shape[1] - 1) // 2)
        word_set = {tuple(word.tolist()) for word in words}
        for word in words:
            for position in range(words.shape[1]):
                received = word.copy()
                received[position] *= -1
                distances_to_code = np.sum(words != received, axis=1)
                valid_single += int(np.sum(distances_to_code == 1) == 1)
            for first, second in itertools.combinations(range(words.shape[1]), 2):
                received = word.copy()
                received[[first, second]] *= -1
                valid_double_detect += int(tuple(received.tolist()) not in word_set)
        rows.append({
            "shortened_by": shortened_by,
            "parameters": f"[{8-shortened_by},{4-shortened_by},{minimum_distance}]",
            "physical_bits": 8 - shortened_by,
            "useful_bits": 4 - shortened_by,
            "code_size": len(words),
            "expected_code_size": 2 ** (4 - shortened_by),
            "minimum_distance": minimum_distance,
            "single_errors_uniquely_correctable": valid_single,
            "single_error_cases": total_single,
            "all_single_errors_correctable": int(valid_single == total_single),
            "double_errors_detected": valid_double_detect,
            "double_error_cases": total_double,
            "all_double_errors_detected": int(valid_double_detect == total_double),
            "removed_full_code_positions": json.dumps(removed_positions),
            "existing_decoder_compatible": int(shortened_by == 0),
        })
    return pd.DataFrame(rows)


def capacity_rows(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for partition in ("development", "validation"):
        cohort = load_partitioned_cohort(
            args.results, args.performance_groups, args.partition, partition
        )
        for row in cohort.itertuples(index=False):
            note_count = int(row.num_notes)
            capacity = nominal_capacity_bits(note_count, N_VALUE)
            base_bits = message_length_for_fraction(capacity, BASE_FRACTION)
            item: dict[str, object] = {
                "experiment_version": EXPERIMENT_VERSION,
                "partition": partition,
                "replay_file": str(row.replay_file),
                "beatmap_hash": str(row.beatmap_hash),
                "performance_category": str(row.performance_category),
                "accuracy": float(row.accuracy),
                "miss_fraction": float(row.miss_fraction),
                "note_index_length": note_count,
                "N": N_VALUE,
                "nominal_coded_bit_capacity": capacity,
                "coded_bits_at_7_5pct": base_bits,
                "complete_secded_words_at_7_5pct": base_bits // CODE_LENGTH,
                "useful_bits_at_7_5pct": (base_bits // CODE_LENGTH) * INFO_LENGTH,
                "eligible_at_7_5pct": int(base_bits >= CODE_LENGTH),
                "structurally_can_fit_full_codeword": int(capacity >= CODE_LENGTH),
            }
            for words in (1, 2, 3):
                item[f"minimum_fraction_{words}_codewords"] = minimum_fraction(
                    capacity, words * CODE_LENGTH
                )
            for cap in CAPS:
                allocated = message_length_for_fraction(capacity, cap)
                item[f"coded_bits_at_{cap:g}"] = allocated
                item[f"eligible_at_{cap:g}"] = int(allocated >= CODE_LENGTH)
            frames.append(item)
    return pd.DataFrame(frames)


def coverage_curve(capacity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition, group in capacity.groupby("partition", sort=False):
        map_count = group.beatmap_hash.nunique()
        for cap in CAPS:
            bits = group[f"coded_bits_at_{cap:g}"].astype(int)
            eligible = bits >= CODE_LENGTH
            required = group.loc[eligible, "minimum_fraction_1_codewords"].dropna()
            eligible_maps = group.loc[eligible, "beatmap_hash"].nunique()
            rows.append({
                "partition": partition,
                "payload_cap": cap,
                "replays": len(group),
                "eligible_replays": int(eligible.sum()),
                "replay_eligibility_fraction": float(eligible.mean()),
                "maps": map_count,
                "eligible_maps": eligible_maps,
                "beatmap_eligibility_fraction": eligible_maps / map_count,
                "median_complete_codewords_per_replay": float((bits // 8).median()),
                "mean_useful_bits_per_replay": float(((bits // 8) * 4).mean()),
                "required_fraction_min_among_eligible": float(required.min()) if len(required) else np.nan,
                "required_fraction_median_among_eligible": float(required.median()) if len(required) else np.nan,
                "required_fraction_p90_among_eligible": float(required.quantile(0.9)) if len(required) else np.nan,
                "unable_fraction": float((~eligible).mean()),
            })
    return pd.DataFrame(rows)


def coverage_by_map(capacity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (partition, beatmap_hash), group in capacity.groupby(["partition", "beatmap_hash"]):
        row: dict[str, object] = {
            "partition": partition, "beatmap_hash": beatmap_hash,
            "replays": len(group), "note_index_length": int(group.note_index_length.iloc[0]),
            "nominal_coded_bit_capacity": int(group.nominal_coded_bit_capacity.iloc[0]),
            "minimum_fraction_1_codewords": float(group.minimum_fraction_1_codewords.iloc[0]),
        }
        for cap in CAPS:
            row[f"eligible_replays_at_{cap:g}"] = int(group[f"eligible_at_{cap:g}"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_rows(capacity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition, group in capacity.groupby("partition", sort=False):
        newly_possible = group[(group.eligible_at_7_5pct == 0) & (group.structurally_can_fit_full_codeword == 1)]
        for scope, scoped in (("all", group), ("newly_possible", newly_possible)):
            values = scoped.minimum_fraction_1_codewords.dropna()
            rows.append({
                "partition": partition, "scope": scope, "replays": len(scoped),
                "minimum": float(values.min()) if len(values) else np.nan,
                "q10": float(values.quantile(.1)) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "q90": float(values.quantile(.9)) if len(values) else np.nan,
                "maximum": float(values.max()) if len(values) else np.nan,
                "structurally_unable": int((scoped.structurally_can_fit_full_codeword == 0).sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "capacity_coverage_v1")
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output nije prazan: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capacity = capacity_rows(args)
    if len(capacity) != 949 or capacity.query("partition == 'development'").shape[0] != 611:
        raise RuntimeError("Frozen cohort mora imati 611 development + 338 validation replay-a.")
    validation = capacity.query("partition == 'validation'")
    if int(validation.eligible_at_7_5pct.sum()) != 167 or validation.loc[validation.eligible_at_7_5pct == 1, "beatmap_hash"].nunique() != 5:
        raise RuntimeError("7.5% audit ne reprodukuje frozen validation eligibility 167/338, 5/11.")
    shortened = shortened_code_rows()
    coverage_short = []
    for partition, group in capacity.groupby("partition", sort=False):
        allocated = group.coded_bits_at_7_5pct.astype(int)
        for code in shortened.itertuples(index=False):
            eligible = allocated >= int(code.physical_bits)
            coverage_short.append({
                "partition": partition, "parameters": code.parameters,
                "physical_bits": code.physical_bits, "useful_bits": code.useful_bits,
                "eligible_replays_at_7_5pct": int(eligible.sum()),
                "coverage_fraction_at_7_5pct": float(eligible.mean()),
                "eligible_maps_at_7_5pct": group.loc[eligible, "beatmap_hash"].nunique(),
            })
    capacity.to_csv(args.output_dir / "capacity_by_replay.csv", index=False)
    coverage_curve(capacity).to_csv(args.output_dir / "coverage_curve.csv", index=False)
    coverage_by_map(capacity).to_csv(args.output_dir / "coverage_by_map.csv", index=False)
    distribution_rows(capacity).to_csv(args.output_dir / "required_payload_distribution.csv", index=False)
    shortened.merge(pd.DataFrame(coverage_short), on=["parameters", "physical_bits", "useful_bits"], how="left").to_csv(
        args.output_dir / "shortened_code_analysis.csv", index=False
    )
    diagnostics = {
        "experiment_version": EXPERIMENT_VERSION,
        "payload_semantics": "C=floor(M/N); B=0 if C<=0 else max(1,floor(C*f)); complete SECDED words=floor(B/8)",
        "missing_carrier_semantics": "NaN/missing carriers do not reduce nominal C; they affect physical reliability only",
        "incomplete_word_semantics": "trailing B mod 8 coded-bit allocation is unused by SECDED",
        "results_sha256": file_sha256(args.results),
        "performance_groups_sha256": file_sha256(args.performance_groups),
        "partition_sha256": file_sha256(args.partition),
        "source_sha256": file_sha256(Path(__file__)),
    }
    (args.output_dir / "diagnostics.csv").write_text(
        pd.DataFrame([diagnostics]).to_csv(index=False), encoding="utf-8"
    )
    print(coverage_curve(capacity).to_string(index=False))


if __name__ == "__main__":
    main()
