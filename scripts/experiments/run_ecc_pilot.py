"""Development-only physical SECDED (8,4) pilot for adaptive DISTRIBUTED."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import decode_hamming_8_4_secded, encode_hamming_8_4_secded
from osu_stego.stego.payload_layout import extract_bipolar_message_with_layout
from scripts.experiments.adaptive_layout_strong_common import (
    LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PAYLOAD_FRACTION, PN_KEY,
    replace_config_rows, write_or_validate_config,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import build_physical_stego
from scripts.experiments.run_payload_sweep import (
    deterministic_message, message_length_for_fraction, nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index, load_map_offsets, prepare_replay, stable_seed,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "ecc-pilot-v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ecc_pilot_v1"
DEFAULT_POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"
DEFAULT_PARTITION = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
RESULT_FIELDS = [
    "experiment_version", "config_id", "replay_file", "beatmap_hash",
    "performance_category", "layout_seed", "layout_key_id", "alpha", "N",
    "payload_fraction", "nominal_payload_bits", "encoded_bits", "useful_info_bits",
    "unused_nominal_bits", "raw_bit_errors", "raw_ber", "post_ecc_bit_errors",
    "post_ecc_ber", "full_message_recovered", "codewords", "clean_codewords",
    "corrected_single_codewords", "detected_double_codewords",
    "post_ecc_failed_codewords", "undetected_or_miscorrected_codewords",
    "requested_carriers", "active_carriers", "dropped_for_hit_window",
    "dropped_for_chronology", "new_unmatched_events", "policy_sha256",
    "replay_sha256",
]


def key_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def select_pilot(cohort: pd.DataFrame, per_group: int = 20) -> pd.DataFrame:
    cohort = cohort.copy()
    cohort["nominal_payload_bits"] = cohort.num_notes.map(
        lambda count: message_length_for_fraction(
            nominal_capacity_bits(int(count), N_VALUE), PAYLOAD_FRACTION
        )
    )
    cohort = cohort[cohort.nominal_payload_bits >= 8]
    selected = []
    for category in ("Very Poor", "Poor", "Good", "Very Good"):
        group = cohort[cohort.performance_category == category].sample(
            frac=1.0,
            random_state=stable_seed(42, EXPERIMENT_VERSION, category),
        )
        first_per_map = group.drop_duplicates("beatmap_hash").head(per_group)
        remaining = group[~group.replay_file.isin(first_per_map.replay_file)]
        chosen = pd.concat(
            [first_per_map, remaining.head(per_group - len(first_per_map))],
            ignore_index=True,
        )
        if len(chosen) != per_group:
            raise ValueError(f"Nema {per_group} eligible replay-eva za {category}.")
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def config_id(replay_file: str, seed: int, replay_sha256: str) -> str:
    material = f"{EXPERIMENT_VERSION}|{replay_file}|{seed}|{replay_sha256}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(pd.read_csv(path, usecols=["config_id"]).config_id.astype(str))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", results)] + [
        (str(seed), group) for seed, group in results.groupby("layout_seed")
    ]
    for seed, group in groups:
        encoded = int(group.encoded_bits.sum())
        useful = int(group.useful_info_bits.sum())
        rows.append({
            "layout_seed": seed,
            "replays": group.replay_file.nunique(),
            "configs": len(group),
            "raw_bits": encoded,
            "raw_bit_errors": int(group.raw_bit_errors.sum()),
            "raw_ber": float(group.raw_bit_errors.sum() / encoded),
            "post_ecc_info_bits": useful,
            "post_ecc_bit_errors": int(group.post_ecc_bit_errors.sum()),
            "post_ecc_ber": float(group.post_ecc_bit_errors.sum() / useful),
            "full_message_recovery_rate": float(group.full_message_recovered.mean()),
            "ecc_overhead_fraction": float((encoded - useful) / encoded),
            "useful_information_bits": useful,
            "effective_useful_bits_per_replay": float(useful / len(group)),
            "nominal_payload_bits_per_replay": float(group.nominal_payload_bits.mean()),
            "effective_useful_fraction_of_nominal_payload": float(
                useful / group.nominal_payload_bits.sum()
            ),
            "unused_nominal_bits": int(group.unused_nominal_bits.sum()),
            "codewords": int(group.codewords.sum()),
            "corrected_single_codewords": int(group.corrected_single_codewords.sum()),
            "detected_double_codewords": int(group.detected_double_codewords.sum()),
            "post_ecc_failed_codewords": int(group.post_ecc_failed_codewords.sum()),
            "undetected_or_miscorrected_codewords": int(group.undetected_or_miscorrected_codewords.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_sender_local_policy(args.policy)
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "development_only_80_replay_physical_pilot",
        "code": "extended_hamming_secded_8_4",
        "layout_seed_count": len(LAYOUT_KEYS),
        "N": N_VALUE,
        "payload_fraction_upper_bound": PAYLOAD_FRACTION,
        "policy_sha256": file_sha256(args.policy),
        "cohort_sha256": file_sha256(args.results),
        "partition_sha256": file_sha256(args.partition),
        "map_offsets_sha256": file_sha256(args.map_offsets),
    }
    write_or_validate_config(args.output_dir / "config.json", config)
    cohort = load_partitioned_cohort(
        args.results, args.performance_groups, args.partition, "development"
    )
    selected = select_pilot(cohort)
    selection_path = args.output_dir / "pilot_selection.csv"
    if selection_path.is_file():
        old = pd.read_csv(selection_path)
        if old.replay_file.tolist() != selected.replay_file.tolist():
            raise ValueError("Postojeći ECC pilot izbor se razlikuje.")
    else:
        selected.to_csv(selection_path, index=False)
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    result_path = args.output_dir / "physical_results.csv"
    done = completed(result_path)
    pending = []
    new = 0
    with tempfile.TemporaryDirectory(prefix="osu_ecc_pilot_") as name:
        temp_dir = Path(name)
        for index, row in enumerate(selected.itertuples(index=False), 1):
            context = prepare_replay(row, args.dataset_dir, beatmaps, offsets)
            replay_hash = file_sha256(context.osr_path)
            alpha = alpha_from_replay(policy, context.replay)
            nominal = message_length_for_fraction(
                nominal_capacity_bits(len(context.original_residuals), N_VALUE),
                PAYLOAD_FRACTION,
            )
            codewords = nominal // 8
            useful_bits = codewords * 4
            encoded_bits = codewords * 8
            info = deterministic_message(
                context.replay_file, context.beatmap_hash, 0.0, N_VALUE,
                PAYLOAD_FRACTION, useful_bits, MESSAGE_SEED,
            )
            encoded = encode_hamming_8_4_secded(info)
            for seed, layout_key in enumerate(LAYOUT_KEYS):
                identifier = config_id(context.replay_file, seed, replay_hash)
                if identifier in done:
                    continue
                _, roundtrip, diagnostics = build_physical_stego(
                    context, encoded, alpha, N_VALUE, "distributed", PN_KEY,
                    layout_key, float(policy["hit_margin_ms"]),
                    temp_dir / f"{identifier}.osr",
                )
                raw = extract_bipolar_message_with_layout(
                    roundtrip, PN_KEY, layout_key, N_VALUE, encoded_bits, "distributed"
                )
                decoded, statuses = decode_hamming_8_4_secded(raw)
                raw_errors = int(np.sum(raw != encoded))
                post_errors = int(np.sum(decoded != info))
                true_words = info.reshape(-1, 4)
                decoded_words = decoded.reshape(-1, 4)
                failed_words = np.any(true_words != decoded_words, axis=1)
                detected = statuses == "detected_double"
                pending.append({
                    "experiment_version": EXPERIMENT_VERSION,
                    "config_id": identifier,
                    "replay_file": context.replay_file,
                    "beatmap_hash": context.beatmap_hash,
                    "performance_category": row.performance_category,
                    "layout_seed": seed,
                    "layout_key_id": key_id(layout_key),
                    "alpha": alpha,
                    "N": N_VALUE,
                    "payload_fraction": PAYLOAD_FRACTION,
                    "nominal_payload_bits": nominal,
                    "encoded_bits": encoded_bits,
                    "useful_info_bits": useful_bits,
                    "unused_nominal_bits": nominal - encoded_bits,
                    "raw_bit_errors": raw_errors,
                    "raw_ber": raw_errors / encoded_bits,
                    "post_ecc_bit_errors": post_errors,
                    "post_ecc_ber": post_errors / useful_bits,
                    "full_message_recovered": int(post_errors == 0),
                    "codewords": codewords,
                    "clean_codewords": int(np.sum(statuses == "clean")),
                    "corrected_single_codewords": int(np.sum(statuses == "corrected_single")),
                    "detected_double_codewords": int(np.sum(detected)),
                    "post_ecc_failed_codewords": int(np.sum(failed_words)),
                    "undetected_or_miscorrected_codewords": int(np.sum(failed_words & ~detected)),
                    "requested_carriers": diagnostics["requested_carriers"],
                    "active_carriers": diagnostics["active_carriers"],
                    "dropped_for_hit_window": diagnostics["dropped_for_hit_window"],
                    "dropped_for_chronology": diagnostics["dropped_for_chronology"],
                    "new_unmatched_events": diagnostics["new_unmatched_events"],
                    "policy_sha256": config["policy_sha256"],
                    "replay_sha256": replay_hash,
                })
                new += 1
                if new % 10 == 0:
                    replace_config_rows(result_path, RESULT_FIELDS, pending)
                    pending = []
            if index % 20 == 0:
                print(f"ECC pilot: {index}/80 replays | new configs={new}", flush=True)
        replace_config_rows(result_path, RESULT_FIELDS, pending)
    results = pd.read_csv(result_path)
    if len(results) != 80 * len(LAYOUT_KEYS):
        raise ValueError("ECC pilot nije kompletan.")
    summarize(results).to_csv(args.output_dir / "ecc_summary.csv", index=False)
    print(f"ECC pilot complete | new configs={new}")


if __name__ == "__main__":
    main()
