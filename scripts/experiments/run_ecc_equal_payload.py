"""Paired physical ECC controls at equal useful and equal physical payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import (
    decode_hamming_8_4_secded,
    encode_hamming_8_4_secded,
    secded_syndrome,
)
from osu_stego.stego.payload_layout import (
    extract_bipolar_message_with_layout,
    message_correlations_with_layout,
)
from scripts.experiments.adaptive_layout_strong_common import (
    LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PAYLOAD_FRACTION, PN_KEY,
    replace_config_rows, write_or_validate_config,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_pilot import DEFAULT_PARTITION, DEFAULT_POLICY
from scripts.experiments.run_layout_comparison import build_physical_stego
from scripts.experiments.run_payload_sweep import (
    deterministic_message, message_length_for_fraction, nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index, load_map_offsets, prepare_replay, stable_seed,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "ecc-equal-payload-v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ecc_equal_payload_v1"
SOURCE_ECC_DIR = RESULTS_DIR / "ecc_pilot_v1"
CONDITIONS = ("ecc", "uncoded_equal_useful", "uncoded_equal_physical")
PHYSICAL_FIELDS = [
    "experiment_version", "config_id", "condition", "replay_file", "beatmap_hash",
    "performance_category", "layout_seed", "layout_key_id", "alpha", "N",
    "payload_fraction", "nominal_payload_bits", "useful_bits", "physical_bits",
    "message_sha256", "useful_message_sha256", "raw_bit_errors", "raw_ber",
    "post_ecc_bit_errors", "post_ecc_ber", "full_message_recovered", "codewords",
    "corrected_single_codewords", "detected_double_codewords",
    "failed_codewords", "undetected_or_miscorrected_codewords",
    "requested_carriers", "active_carriers", "active_carrier_fraction",
    "dropped_for_hit_window", "dropped_for_chronology", "new_unmatched_events",
    "new_matched_events", "changed_match_status_total", "sum_squared_shift_ms2",
    "rms_applied_shift_ms", "mean_absolute_applied_shift_ms", "policy_sha256",
    "replay_sha256", "pn_key_id", "source_ecc_results_sha256",
]
BIT_FIELDS = [
    "config_id", "replay_file", "beatmap_hash", "layout_seed", "alpha",
    "bit_index", "codeword_index", "codeword_position", "encoded_bit",
    "hard_decoded_bit", "is_raw_error", "correlation", "abs_correlation",
]
WORD_FIELDS = [
    "config_id", "replay_file", "beatmap_hash", "layout_seed", "alpha",
    "codeword_index", "true_info_bits", "encoded_bits", "hard_received_bits",
    "hard_decoded_info_bits", "raw_error_count", "raw_error_positions",
    "hard_status", "syndrome", "overall_parity", "hard_failed",
    "silent_miscorrection", "abs_correlations", "wrong_bit_confidence_ranks",
]


def key_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int8).tobytes()).hexdigest()


def identifier(
    condition: str,
    replay_file: str,
    replay_sha256: str,
    seed: int,
    message: np.ndarray,
    policy_sha256: str,
) -> str:
    material = "|".join((
        EXPERIMENT_VERSION, condition, replay_file, replay_sha256, str(seed),
        array_hash(message), policy_sha256, key_id(PN_KEY), key_id(LAYOUT_KEYS[seed]),
        str(N_VALUE), str(PAYLOAD_FRACTION),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def completed_configs(physical_path: Path, bit_path: Path, word_path: Path) -> set[str]:
    if not physical_path.is_file():
        return set()
    physical = pd.read_csv(physical_path)
    complete = set(
        physical.loc[physical.condition != "ecc", "config_id"].astype(str)
    )
    if bit_path.is_file() and word_path.is_file():
        bits = pd.read_csv(bit_path)
        words = pd.read_csv(word_path)
        for _, row in physical[physical.condition == "ecc"].iterrows():
            config_id = str(row.config_id)
            if (
                int((bits.config_id == config_id).sum()) == int(row.physical_bits)
                and int((words.config_id == config_id).sum()) == int(row.codewords)
            ):
                complete.add(config_id)
    return complete


def cluster_bootstrap_delta(
    paired: pd.DataFrame,
    candidate: str,
    control: str,
    metric: str,
    iterations: int = 5000,
) -> tuple[float, float, float]:
    wide = paired.pivot(
        index=["replay_file", "layout_seed"], columns="condition", values=metric
    ).reset_index()
    replay_ids = wide.replay_file.drop_duplicates().to_numpy()
    delta = float((wide[candidate] - wide[control]).mean())
    rng = np.random.default_rng(stable_seed(42, EXPERIMENT_VERSION, candidate, control, metric))
    replay_delta = (
        wide.assign(delta=wide[candidate] - wide[control])
        .groupby("replay_file")["delta"]
        .mean()
        .reindex(replay_ids)
        .to_numpy(float)
    )
    indices = rng.integers(0, len(replay_delta), size=(iterations, len(replay_delta)))
    samples = replay_delta[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return delta, float(low), float(high)


def analyze(
    results: pd.DataFrame,
    output_dir: Path,
    expected_replays: int,
    word_path: Path,
) -> None:
    expected = expected_replays * len(LAYOUT_KEYS) * len(CONDITIONS)
    if len(results) != expected or results.config_id.nunique() != expected:
        raise ValueError("Equal-payload fizički rezultat nije kompletan/jedinstven.")
    counts = results.groupby(["replay_file", "layout_seed"]).condition.nunique()
    if not (counts == len(CONDITIONS)).all():
        raise ValueError("Nedostaje paired condition.")
    words = pd.read_csv(word_path)
    detected_configs = set(
        words.loc[words.hard_status == "detected_double", "config_id"].astype(str)
    )
    results = results.copy()
    results["operational_full_message_recovered"] = results[
        "full_message_recovered"
    ].astype(int)
    ecc_mask = results.condition == "ecc"
    results.loc[ecc_mask, "operational_full_message_recovered"] = (
        results.loc[ecc_mask, "full_message_recovered"].astype(bool)
        & ~results.loc[ecc_mask, "config_id"].astype(str).isin(detected_configs)
    ).astype(int)
    summary = []
    for condition, group in results.groupby("condition"):
        useful = int(group.useful_bits.sum())
        physical = int(group.physical_bits.sum())
        summary.append({
            "condition": condition, "configs": len(group),
            "replays": group.replay_file.nunique(),
            "useful_bits_per_config": float(group.useful_bits.mean()),
            "physical_bits_per_config": float(group.physical_bits.mean()),
            "useful_bits_per_physical_bit": useful / physical,
            "useful_payload_rate_of_nominal": useful / group.nominal_payload_bits.sum(),
            "raw_ber": group.raw_bit_errors.sum() / physical,
            "post_ecc_ber": group.post_ecc_bit_errors.sum() / useful,
            "full_message_recovery": float(group.full_message_recovered.mean()),
            "operational_declared_recovery": float(
                group.operational_full_message_recovered.mean()
            ),
            "successfully_recovered_full_message_bits_per_config": float(
                (group.useful_bits * group.full_message_recovered).mean()
            ),
            "operationally_recovered_full_message_bits_per_config": float(
                (
                    group.useful_bits
                    * group.operational_full_message_recovered
                ).mean()
            ),
        })
    pd.DataFrame(summary).to_csv(output_dir / "equal_payload_summary.csv", index=False)

    paired_rows = []
    for control, comparison in (
        ("uncoded_equal_useful", "equal_useful_payload"),
        ("uncoded_equal_physical", "equal_physical_budget"),
    ):
        for metric in (
            "full_message_recovered", "operational_full_message_recovered"
        ):
            delta, low, high = cluster_bootstrap_delta(
                results, "ecc", control, metric
            )
            paired_rows.append({
                "comparison": comparison, "recovery_definition": metric,
                "ecc_recovery": float(results[results.condition == "ecc"][metric].mean()),
                "control_recovery": float(results[results.condition == control][metric].mean()),
                "delta_ecc_minus_control": delta, "ci95_low": low, "ci95_high": high,
                "bootstrap_unit": "replay_file", "bootstrap_iterations": 5000,
            })
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_recovery.csv", index=False)

    outcome_rows = []
    for metric in ("full_message_recovered", "operational_full_message_recovered"):
        wide = results.pivot(
            index=["replay_file", "layout_seed"], columns="condition", values=metric
        ).reset_index()
        ecc = wide.ecc.astype(bool)
        uncoded = wide.uncoded_equal_useful.astype(bool)
        outcome_rows.extend([
            {"recovery_definition": metric, "outcome": "both_succeed", "configs": int((ecc & uncoded).sum())},
            {"recovery_definition": metric, "outcome": "ecc_succeeds_uncoded_fails", "configs": int((ecc & ~uncoded).sum())},
            {"recovery_definition": metric, "outcome": "uncoded_succeeds_ecc_fails", "configs": int((~ecc & uncoded).sum())},
            {"recovery_definition": metric, "outcome": "both_fail", "configs": int((~ecc & ~uncoded).sum())},
        ])
    outcomes = pd.DataFrame(outcome_rows)
    outcomes.to_csv(output_dir / "paired_outcomes.csv", index=False)

    lengths = []
    equal_useful = results[
        results.condition.isin(("ecc", "uncoded_equal_useful"))
    ]
    for useful_bits, group in equal_useful.groupby("useful_bits"):
        pivot = group.pivot(
            index=["replay_file", "layout_seed"], columns="condition",
            values="full_message_recovered",
        )
        lengths.append({
            "useful_bits": int(useful_bits), "configs": len(pivot),
            "replays": pivot.reset_index().replay_file.nunique(),
            "ecc_recovery": float(pivot.ecc.mean()),
            "uncoded_equal_useful_recovery": float(pivot.uncoded_equal_useful.mean()),
            "delta": float((pivot.ecc - pivot.uncoded_equal_useful).mean()),
            "ecc_operational_recovery": float(
                group[group.condition == "ecc"].operational_full_message_recovered.mean()
            ),
        })
    pd.DataFrame(lengths).to_csv(output_dir / "recovery_by_message_length.csv", index=False)

    physical_metrics = (
        "active_carriers", "active_carrier_fraction", "sum_squared_shift_ms2",
        "rms_applied_shift_ms", "mean_absolute_applied_shift_ms",
        "dropped_for_chronology", "dropped_for_hit_window", "new_unmatched_events",
        "changed_match_status_total",
    )
    costs = []
    for control in ("uncoded_equal_useful", "uncoded_equal_physical"):
        for metric in physical_metrics:
            delta, low, high = cluster_bootstrap_delta(results, "ecc", control, metric)
            costs.append({
                "control": control, "metric": metric,
                "ecc_value": float(results[results.condition == "ecc"][metric].mean()),
                "control_value": float(results[results.condition == control][metric].mean()),
                "delta_ecc_minus_control": delta, "ci95_low": low, "ci95_high": high,
                "bootstrap_unit": "replay_file",
            })
    pd.DataFrame(costs).to_csv(output_dir / "physical_cost_paired.csv", index=False)

    seeds = results.groupby(["condition", "layout_seed"]).agg(
        configs=("config_id", "size"), raw_errors=("raw_bit_errors", "sum"),
        physical_bits=("physical_bits", "sum"), post_errors=("post_ecc_bit_errors", "sum"),
        useful_bits=("useful_bits", "sum"), recovery=("full_message_recovered", "mean"),
        operational_recovery=("operational_full_message_recovered", "mean"),
    ).reset_index()
    seeds["raw_ber"] = seeds.raw_errors / seeds.physical_bits
    seeds["post_ecc_ber"] = seeds.post_errors / seeds.useful_bits
    seeds.to_csv(output_dir / "seed_summary.csv", index=False)
    pd.DataFrame([
        {"diagnostic": "physical_rows", "value": len(results)},
        {"diagnostic": "unique_configs", "value": results.config_id.nunique()},
        {"diagnostic": "replays", "value": results.replay_file.nunique()},
        {"diagnostic": "conditions_per_pair_min", "value": int(counts.min())},
        {"diagnostic": "conditions_per_pair_max", "value": int(counts.max())},
    ]).to_csv(output_dir / "diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-ecc-dir", type=Path, default=SOURCE_ECC_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--limit-replays", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.analyze_only:
        raise ValueError("Ne mogu zajedno --generate-only i --analyze-only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.source_ecc_dir / "pilot_selection.csv"
    source_physical_path = args.source_ecc_dir / "physical_results.csv"
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "comparison_A": "equal useful K: ECC 2K physical vs uncoded K physical",
        "comparison_B": "equal physical 2K: ECC K useful vs uncoded 2K useful",
        "policy_sha256": file_sha256(args.policy),
        "selection_sha256": file_sha256(selection_path),
        "source_ecc_results_sha256": file_sha256(source_physical_path),
        "cohort_sha256": file_sha256(args.results),
        "partition_sha256": file_sha256(args.partition),
        "map_offsets_sha256": file_sha256(args.map_offsets),
        "N": N_VALUE, "payload_fraction": PAYLOAD_FRACTION,
        "pn_key_id": key_id(PN_KEY),
        "layout_key_ids": [key_id(value) for value in LAYOUT_KEYS],
        "message_seed": MESSAGE_SEED, "limit_replays": args.limit_replays,
    }
    write_or_validate_config(args.output_dir / "config.json", config)
    physical_path = args.output_dir / "physical_results.csv"
    bit_path = args.output_dir / "ecc_bit_results.csv"
    word_path = args.output_dir / "ecc_codeword_results.csv"
    if not args.analyze_only:
        selection = pd.read_csv(selection_path)
        if args.limit_replays is not None:
            selection = selection.head(args.limit_replays)
        cohort = load_partitioned_cohort(
            args.results, args.performance_groups, args.partition, "development"
        )
        selected = selection[["replay_file"]].merge(
            cohort, on="replay_file", how="left", validate="one_to_one"
        )
        if selected.beatmap_hash.isna().any():
            raise ValueError("ECC pilot selection nije podskup frozen development cohort-a.")
        source_ecc = pd.read_csv(source_physical_path).set_index(
            ["replay_file", "layout_seed"], drop=False
        )
        complete = completed_configs(physical_path, bit_path, word_path)
        beatmaps = build_beatmap_index(args.dataset_dir)
        offsets = load_map_offsets(args.map_offsets)
        policy = load_sender_local_policy(args.policy)
        pending_physical: list[dict] = []
        pending_bits: list[dict] = []
        pending_words: list[dict] = []
        new = 0
        with tempfile.TemporaryDirectory(prefix="osu_ecc_equal_") as name:
            temp_dir = Path(name)
            for replay_index, row in enumerate(selected.itertuples(index=False), 1):
                context = prepare_replay(row, args.dataset_dir, beatmaps, offsets)
                replay_hash = file_sha256(context.osr_path)
                alpha = alpha_from_replay(policy, context.replay)
                nominal = message_length_for_fraction(
                    nominal_capacity_bits(len(context.original_residuals), N_VALUE),
                    PAYLOAD_FRACTION,
                )
                useful_count = (nominal // 8) * 4
                physical_count = useful_count * 2
                useful = deterministic_message(
                    context.replay_file, context.beatmap_hash, 0.0, N_VALUE,
                    PAYLOAD_FRACTION, useful_count, MESSAGE_SEED,
                )
                extended = deterministic_message(
                    context.replay_file, context.beatmap_hash, 0.0, N_VALUE,
                    PAYLOAD_FRACTION, physical_count, MESSAGE_SEED,
                )
                if not np.array_equal(extended[:useful_count], useful):
                    raise ValueError("Uncoded useful poruka nije isti deterministic prefix.")
                encoded = encode_hamming_8_4_secded(useful)
                messages = {
                    "ecc": encoded,
                    "uncoded_equal_useful": useful,
                    "uncoded_equal_physical": extended,
                }
                for seed, layout_key in enumerate(LAYOUT_KEYS):
                    source_row = source_ecc.loc[(context.replay_file, seed)]
                    if float(source_row.alpha) != alpha or int(source_row.useful_info_bits) != useful_count:
                        raise ValueError("Novi paired config ne reprodukuje source ECC provenance.")
                    for condition, message in messages.items():
                        config_id = identifier(
                            condition, context.replay_file, replay_hash, seed,
                            message, config["policy_sha256"],
                        )
                        if config_id in complete:
                            continue
                        _, roundtrip, diagnostics = build_physical_stego(
                            context, message, alpha, N_VALUE, "distributed", PN_KEY,
                            layout_key, float(policy["hit_margin_ms"]),
                            temp_dir / f"{config_id}.osr",
                        )
                        raw = extract_bipolar_message_with_layout(
                            roundtrip, PN_KEY, layout_key, N_VALUE, len(message), "distributed"
                        )
                        correlations = message_correlations_with_layout(
                            roundtrip, PN_KEY, layout_key, N_VALUE, len(message), "distributed"
                        )
                        raw_errors = int(np.sum(raw != message))
                        useful_bits = useful_count if condition != "uncoded_equal_physical" else physical_count
                        post_errors = raw_errors
                        statuses = np.asarray([], dtype=str)
                        decoded_info = raw
                        failed_words = np.asarray([], dtype=bool)
                        if condition == "ecc":
                            decoded_info, statuses = decode_hamming_8_4_secded(raw)
                            post_errors = int(np.sum(decoded_info != useful))
                            failed_words = np.any(
                                decoded_info.reshape(-1, 4) != useful.reshape(-1, 4), axis=1
                            )
                            if (
                                raw_errors != int(source_row.raw_bit_errors)
                                or post_errors != int(source_row.post_ecc_bit_errors)
                            ):
                                raise ValueError("Regenerisani ECC ne reprodukuje source pilot.")
                            for bit_index in range(len(message)):
                                pending_bits.append({
                                    "config_id": config_id, "replay_file": context.replay_file,
                                    "beatmap_hash": context.beatmap_hash, "layout_seed": seed,
                                    "alpha": alpha, "bit_index": bit_index,
                                    "codeword_index": bit_index // 8,
                                    "codeword_position": bit_index % 8,
                                    "encoded_bit": int(message[bit_index]),
                                    "hard_decoded_bit": int(raw[bit_index]),
                                    "is_raw_error": int(raw[bit_index] != message[bit_index]),
                                    "correlation": float(correlations[bit_index]),
                                    "abs_correlation": float(abs(correlations[bit_index])),
                                })
                            for word_index in range(len(message) // 8):
                                sl = slice(word_index * 8, (word_index + 1) * 8)
                                info_sl = slice(word_index * 4, (word_index + 1) * 4)
                                errors = np.flatnonzero(raw[sl] != message[sl])
                                syndrome, parity = secded_syndrome(raw[sl])
                                ranks = np.argsort(np.argsort(np.abs(correlations[sl]), kind="stable"), kind="stable")
                                pending_words.append({
                                    "config_id": config_id, "replay_file": context.replay_file,
                                    "beatmap_hash": context.beatmap_hash, "layout_seed": seed,
                                    "alpha": alpha, "codeword_index": word_index,
                                    "true_info_bits": json.dumps(useful[info_sl].tolist()),
                                    "encoded_bits": json.dumps(message[sl].tolist()),
                                    "hard_received_bits": json.dumps(raw[sl].tolist()),
                                    "hard_decoded_info_bits": json.dumps(decoded_info[info_sl].tolist()),
                                    "raw_error_count": len(errors),
                                    "raw_error_positions": json.dumps(errors.tolist()),
                                    "hard_status": str(statuses[word_index]),
                                    "syndrome": syndrome, "overall_parity": parity,
                                    "hard_failed": int(failed_words[word_index]),
                                    "silent_miscorrection": int(
                                        failed_words[word_index] and statuses[word_index] != "detected_double"
                                    ),
                                    "abs_correlations": json.dumps(np.abs(correlations[sl]).tolist()),
                                    "wrong_bit_confidence_ranks": json.dumps(ranks[errors].tolist()),
                                })
                        common = {
                            "experiment_version": EXPERIMENT_VERSION, "config_id": config_id,
                            "condition": condition, "replay_file": context.replay_file,
                            "beatmap_hash": context.beatmap_hash,
                            "performance_category": row.performance_category,
                            "layout_seed": seed, "layout_key_id": key_id(layout_key),
                            "alpha": alpha, "N": N_VALUE, "payload_fraction": PAYLOAD_FRACTION,
                            "nominal_payload_bits": nominal, "useful_bits": useful_bits,
                            "physical_bits": len(message), "message_sha256": array_hash(message),
                            "useful_message_sha256": array_hash(useful),
                            "raw_bit_errors": raw_errors, "raw_ber": raw_errors / len(message),
                            "post_ecc_bit_errors": post_errors,
                            "post_ecc_ber": post_errors / useful_bits,
                            "full_message_recovered": int(post_errors == 0),
                            "codewords": len(message) // 8 if condition == "ecc" else 0,
                            "corrected_single_codewords": int(np.sum(statuses == "corrected_single")),
                            "detected_double_codewords": int(np.sum(statuses == "detected_double")),
                            "failed_codewords": int(np.sum(failed_words)),
                            "undetected_or_miscorrected_codewords": int(
                                np.sum(failed_words & (statuses != "detected_double"))
                            ) if condition == "ecc" else 0,
                            "active_carrier_fraction": (
                                diagnostics["active_carriers"] / diagnostics["requested_carriers"]
                                if diagnostics["requested_carriers"] else 0.0
                            ),
                            "policy_sha256": config["policy_sha256"],
                            "replay_sha256": replay_hash, "pn_key_id": key_id(PN_KEY),
                            "source_ecc_results_sha256": config["source_ecc_results_sha256"],
                            **diagnostics,
                        }
                        pending_physical.append(common)
                        new += 1
                        if new % 15 == 0:
                            replace_config_rows(physical_path, PHYSICAL_FIELDS, pending_physical)
                            replace_config_rows(bit_path, BIT_FIELDS, pending_bits)
                            replace_config_rows(word_path, WORD_FIELDS, pending_words)
                            pending_physical, pending_bits, pending_words = [], [], []
                if replay_index % 10 == 0 or replay_index == len(selected):
                    print(f"equal-payload: {replay_index}/{len(selected)} | new configs={new}", flush=True)
            replace_config_rows(physical_path, PHYSICAL_FIELDS, pending_physical)
            replace_config_rows(bit_path, BIT_FIELDS, pending_bits)
            replace_config_rows(word_path, WORD_FIELDS, pending_words)
        print(f"equal-payload generation complete | new configs={new}")
    if not args.generate_only:
        analyze(
            pd.read_csv(physical_path),
            args.output_dir,
            80 if args.limit_replays is None else args.limit_replays,
            word_path,
        )
        print("equal-payload analysis complete")


if __name__ == "__main__":
    main()
