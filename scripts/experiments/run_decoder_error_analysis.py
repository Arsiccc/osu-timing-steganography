"""Reconstruct and analyze bit-level errors for adaptive DISTRIBUTED embedding."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from osu_stego.analysis.decoder_errors import bit_level_diagnostics, error_runs
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from scripts.experiments.adaptive_layout_strong_common import (
    LAYOUT_KEYS,
    MESSAGE_SEED,
    N_VALUE,
    PAYLOAD_FRACTION,
    PN_KEY,
    replace_config_rows,
    write_or_validate_config,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import build_physical_stego
from scripts.experiments.run_payload_sweep import (
    deterministic_message,
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    prepare_replay,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "decoder-error-analysis-v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "decoder_error_analysis_v1"
DEFAULT_POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"
DEFAULT_PARTITION = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
SOURCE_RESULTS = RESULTS_DIR / "adaptive_layout_strong_v1" / "physical_results.csv"

BIT_FIELDS = [
    "experiment_version", "config_id", "source_config_id", "partition",
    "replay_file", "beatmap_hash", "layout_seed", "layout_key_id", "alpha",
    "n_frames_per_bit", "payload_fraction", "message_bits", "bit_index",
    "true_bit", "decoded_bit", "is_error", "correlation", "abs_correlation",
    "normalized_abs_correlation", "block_index", "block_start_note",
    "block_end_note_exclusive", "block_center_fraction", "block_quarter",
    "roundtrip_valid_carriers", "requested_carriers", "active_carriers",
    "dropped_carriers", "hit_window_drops", "chronology_drops",
    "new_unmatched_events", "new_matched_events", "rematching_changes",
    "clean_valid_residuals", "clean_mean_residual", "clean_variance",
    "clean_mean_absolute_residual", "clean_pn_correlation", "policy_sha256",
    "source_results_sha256", "replay_sha256",
]


def key_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def analysis_config(args: argparse.Namespace) -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "source": "physical_roundtrip_reconstruction_of_existing_adaptive_distributed_configs",
        "policy_version": "adaptive-alpha-sender-local-v2",
        "policy_sha256": file_sha256(args.policy),
        "source_results_sha256": file_sha256(args.source_results),
        "partition_sha256": file_sha256(args.partition),
        "performance_groups_sha256": file_sha256(args.performance_groups),
        "cohort_sha256": file_sha256(args.results),
        "map_offsets_sha256": file_sha256(args.map_offsets),
        "N": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "message_seed": MESSAGE_SEED,
        "pn_key_id": key_id(PN_KEY),
        "layout_keys": [key_id(value) for value in LAYOUT_KEYS],
        "limit_per_partition": args.limit_per_partition,
    }


def completed_configs(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, usecols=["config_id", "bit_index", "message_bits"])
    complete: set[str] = set()
    for config_id, group in frame.groupby("config_id"):
        expected = int(group["message_bits"].iloc[0])
        if len(group) == expected and set(group["bit_index"].astype(int)) == set(range(expected)):
            complete.add(str(config_id))
    return complete


def expected_source() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_RESULTS if not hasattr(expected_source, "path") else expected_source.path)
    source = source[
        (source["status"] == "OK")
        & (source["alpha_method"] == "sender_local_adaptive")
        & (source["layout"] == "distributed")
    ].copy()
    return source.set_index(["partition", "replay_file", "layout_seed"], drop=False)


def validate_totals(rows: list[dict], totals: dict[str, int], expected: pd.Series) -> None:
    if sum(int(row["is_error"]) for row in rows) != int(expected.bit_errors_roundtrip):
        raise ValueError("Bit-level BER ne reprodukuje frozen physical rezultat.")
    for name in (
        "requested_carriers", "active_carriers", "dropped_for_hit_window",
        "dropped_for_chronology", "new_unmatched_events", "new_matched_events",
        "changed_match_status_total",
    ):
        if int(totals[name]) != int(expected[name]):
            raise ValueError(f"Bit-level {name} ne reprodukuje frozen rezultat.")


def generate(args: argparse.Namespace, config: dict) -> None:
    output = args.output_dir / "bit_level_results.csv"
    complete = completed_configs(output)
    source = expected_source()
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    policy = load_sender_local_policy(args.policy)
    pending: list[dict] = []
    new_configs = 0
    with tempfile.TemporaryDirectory(prefix="osu_decoder_errors_") as name:
        temp_dir = Path(name)
        for partition_name in ("development", "validation"):
            cohort = load_partitioned_cohort(
                args.results, args.performance_groups, args.partition,
                partition_name, args.limit_per_partition,
            )
            for replay_index, row in enumerate(cohort.itertuples(index=False), 1):
                context = prepare_replay(row, args.dataset_dir, beatmaps, offsets)
                alpha = alpha_from_replay(policy, context.replay)
                capacity = nominal_capacity_bits(len(context.original_residuals), N_VALUE)
                num_bits = message_length_for_fraction(capacity, PAYLOAD_FRACTION)
                message = deterministic_message(
                    context.replay_file, context.beatmap_hash, 0.0, N_VALUE,
                    PAYLOAD_FRACTION, num_bits, MESSAGE_SEED,
                )
                for seed_index, layout_key in enumerate(LAYOUT_KEYS):
                    expected = source.loc[(partition_name, context.replay_file, str(seed_index))]
                    config_id = str(expected.config_id)
                    if config_id in complete:
                        continue
                    if alpha != float(expected.alpha) or num_bits != int(expected.message_bits):
                        raise ValueError("V2 alpha/message length menja frozen config.")
                    _, roundtrip, diagnostics = build_physical_stego(
                        context, message, alpha, N_VALUE, "distributed", PN_KEY,
                        layout_key, float(policy["hit_margin_ms"]),
                        temp_dir / f"{config_id}.osr",
                    )
                    bit_rows, totals = bit_level_diagnostics(
                        context=context,
                        roundtrip_residuals=roundtrip,
                        message=message,
                        alpha=alpha,
                        n_frames_per_bit=N_VALUE,
                        pn_key=PN_KEY,
                        layout_key=layout_key,
                        hit_margin_ms=float(policy["hit_margin_ms"]),
                    )
                    validate_totals(bit_rows, totals, expected)
                    if int(diagnostics["changed_match_status_total"]) != totals["changed_match_status_total"]:
                        raise ValueError("Reconstructed diagnostics nisu interno konzistentne.")
                    common = {
                        "experiment_version": EXPERIMENT_VERSION,
                        "config_id": config_id,
                        "source_config_id": config_id,
                        "partition": partition_name,
                        "replay_file": context.replay_file,
                        "beatmap_hash": context.beatmap_hash,
                        "layout_seed": seed_index,
                        "layout_key_id": key_id(layout_key),
                        "alpha": alpha,
                        "n_frames_per_bit": N_VALUE,
                        "payload_fraction": PAYLOAD_FRACTION,
                        "message_bits": num_bits,
                        "policy_sha256": config["policy_sha256"],
                        "source_results_sha256": config["source_results_sha256"],
                        "replay_sha256": file_sha256(context.osr_path),
                    }
                    pending.extend({**common, **bit_row} for bit_row in bit_rows)
                    new_configs += 1
                    if new_configs % 10 == 0:
                        replace_config_rows(output, BIT_FIELDS, pending)
                        pending = []
                if replay_index % 25 == 0 or replay_index == len(cohort):
                    print(
                        f"bit reconstruction {partition_name}: {replay_index}/{len(cohort)} "
                        f"| new configs={new_configs}",
                        flush=True,
                    )
        replace_config_rows(output, BIT_FIELDS, pending)
    print(f"bit reconstruction complete | new configs={new_configs}")


def rate_row(scope: str, value: str, frame: pd.DataFrame) -> dict:
    return {
        "scope": scope, "group_value": value, "total_bits": len(frame),
        "bit_errors": int(frame.is_error.sum()), "error_rate": float(frame.is_error.mean()),
        "metric": "", "value": np.nan,
    }


def independence_rows(bits: pd.DataFrame, scope: str, value: str) -> list[dict]:
    pairs = []
    for _, group in bits.groupby("config_id"):
        errors = group.sort_values("bit_index").is_error.to_numpy(dtype=np.float64)
        if len(errors) > 1:
            pairs.append(np.column_stack((errors[:-1], errors[1:])))
    pair = np.concatenate(pairs) if pairs else np.empty((0, 2))
    p = float(bits.is_error.mean())
    observed = float(np.mean((pair[:, 0] == 1) & (pair[:, 1] == 1)))
    lag1 = float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1]) if pair.size and np.std(pair[:, 0]) and np.std(pair[:, 1]) else np.nan
    metrics = {
        "lag1_error_autocorrelation": lag1,
        "observed_consecutive_error_pair_frequency": observed,
        "independence_expected_consecutive_error_pair_frequency": p * p,
        "observed_to_expected_consecutive_ratio": observed / (p * p) if p else np.nan,
        "adjacent_bit_pairs": float(len(pair)),
    }
    return [
        {"scope": scope, "group_value": value, "total_bits": len(bits),
         "bit_errors": int(bits.is_error.sum()), "error_rate": p,
         "metric": metric, "value": number}
        for metric, number in metrics.items()
    ]


def replay_conditioned_independence_rows(bits: pd.DataFrame) -> list[dict]:
    """Contrast adjacent errors with an independence model conditional on replay."""
    replay_rates = bits.groupby("replay_file")["is_error"].mean()
    observed_pairs = 0
    expected_pairs = 0.0
    adjacent_pairs = 0
    for (replay_file, _), group in bits.groupby(["replay_file", "layout_seed"]):
        errors = group.sort_values("bit_index")["is_error"].to_numpy(dtype=np.int8)
        pair_count = max(0, len(errors) - 1)
        adjacent_pairs += pair_count
        observed_pairs += int(np.sum(errors[:-1] * errors[1:]))
        expected_pairs += pair_count * float(replay_rates.loc[replay_file]) ** 2
    metrics = {
        "observed_consecutive_error_pairs": float(observed_pairs),
        "replay_conditioned_independence_expected_pairs_plugin": expected_pairs,
        "observed_to_replay_conditioned_expected_ratio": (
            observed_pairs / expected_pairs if expected_pairs else np.nan
        ),
        "adjacent_bit_pairs": float(adjacent_pairs),
    }
    return [
        {
            "scope": "replay_conditioned_independence",
            "group_value": "all",
            "total_bits": len(bits),
            "bit_errors": int(bits.is_error.sum()),
            "error_rate": float(bits.is_error.mean()),
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def build_runs(bits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, group in bits.groupby("config_id"):
        group = group.sort_values("bit_index")
        first = group.iloc[0]
        for start, length in error_runs(group.is_error.to_numpy(dtype=np.int8)):
            rows.append({
                "config_id": config_id, "partition": first.partition,
                "replay_file": first.replay_file, "beatmap_hash": first.beatmap_hash,
                "layout_seed": int(first.layout_seed), "start_bit_index": start,
                "run_length": length,
            })
    return pd.DataFrame(rows, columns=[
        "config_id", "partition", "replay_file", "beatmap_hash", "layout_seed",
        "start_bit_index", "run_length",
    ])


def confidence_bins(bits: pd.DataFrame) -> pd.DataFrame:
    ranked = bits.copy()
    ranked["confidence_bin"] = pd.qcut(
        ranked.abs_correlation.rank(method="first"), 10, labels=False
    ).astype(int)
    rows = []
    for index, group in ranked.groupby("confidence_bin"):
        rows.append({
            "bin_index": int(index), "min_abs_correlation": float(group.abs_correlation.min()),
            "max_abs_correlation": float(group.abs_correlation.max()), "bits": len(group),
            "errors": int(group.is_error.sum()), "error_probability": float(group.is_error.mean()),
            "mean_abs_correlation": float(group.abs_correlation.mean()),
        })
    return pd.DataFrame(rows)


def carrier_rows(bits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    factors = (
        "roundtrip_valid_carriers", "active_carriers", "dropped_carriers",
        "hit_window_drops", "chronology_drops", "rematching_changes",
    )
    for factor in factors:
        for value, group in bits.groupby(factor):
            rows.append({
                "row_type": "group", "factor": factor, "factor_value": value,
                "bits": len(group), "errors": int(group.is_error.sum()),
                "error_rate": float(group.is_error.mean()),
                "mean_abs_correlation": float(group.abs_correlation.mean()),
                "predictor_auc": np.nan, "spearman_error": np.nan,
            })
    predictor_scores = {
        "low_abs_correlation": -bits.abs_correlation,
        "few_roundtrip_valid_carriers": -bits.roundtrip_valid_carriers,
        "many_dropped_carriers": bits.dropped_carriers,
        "many_hit_window_drops": bits.hit_window_drops,
        "many_chronology_drops": bits.chronology_drops,
        "many_rematching_changes": bits.rematching_changes,
        "high_clean_variance": bits.clean_variance,
        "high_clean_mean_absolute_residual": bits.clean_mean_absolute_residual,
        "high_clean_abs_pn_correlation": bits.clean_pn_correlation.abs(),
    }
    for factor, score in predictor_scores.items():
        valid = np.isfinite(score.to_numpy(float))
        auc = roc_auc_score(bits.is_error[valid], score[valid])
        correlation = spearmanr(score[valid], bits.is_error[valid]).statistic
        rows.append({
            "row_type": "predictor", "factor": factor, "factor_value": "",
            "bits": int(valid.sum()), "errors": int(bits.is_error[valid].sum()),
            "error_rate": float(bits.is_error[valid].mean()),
            "mean_abs_correlation": np.nan, "predictor_auc": float(auc),
            "spearman_error": float(correlation),
        })
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> None:
    bits = pd.read_csv(args.output_dir / "bit_level_results.csv")
    source = expected_source().reset_index(drop=True)
    expected_configs = 5 * (949 if args.limit_per_partition is None else 2 * args.limit_per_partition)
    if bits.config_id.nunique() != expected_configs:
        raise ValueError("Bit-level output nema očekivan broj kompletnih config-a.")
    if int(bits.is_error.sum()) != int(source[source.config_id.isin(bits.config_id)].bit_errors_roundtrip.sum()):
        raise ValueError("Globalni bit errors ne reprodukuje source rezultate.")
    summary = [rate_row("overall", "all", bits)]
    for seed, group in bits.groupby("layout_seed"):
        summary.append(rate_row("layout_seed", str(seed), group))
    for quarter, group in bits.groupby("block_quarter"):
        summary.append(rate_row("quarter", str(quarter), group))
    for bit_index, group in bits.groupby("bit_index"):
        summary.append(rate_row("bit_index", str(bit_index), group))
    for replay_file, group in bits.groupby("replay_file"):
        summary.append(rate_row("replay", str(replay_file), group))
    summary.extend(independence_rows(bits, "overall", "all"))
    summary.extend(replay_conditioned_independence_rows(bits))
    for seed, group in bits.groupby("layout_seed"):
        summary.extend(independence_rows(group, "layout_seed_independence", str(seed)))
    correct = bits[bits.is_error == 0].abs_correlation
    error = bits[bits.is_error == 1].abs_correlation
    confidence_auc = roc_auc_score(bits.is_error, -bits.abs_correlation)
    for metric, value in {
        "median_abs_correlation_correct": correct.median(),
        "median_abs_correlation_error": error.median(),
        "mean_abs_correlation_correct": correct.mean(),
        "mean_abs_correlation_error": error.mean(),
        "low_abs_correlation_error_auc": confidence_auc,
    }.items():
        summary.append({
            "scope": "decoder_confidence", "group_value": "all", "total_bits": len(bits),
            "bit_errors": int(bits.is_error.sum()), "error_rate": float(bits.is_error.mean()),
            "metric": metric, "value": float(value),
        })
    runs = build_runs(bits)
    for length, count in runs.run_length.value_counts().sort_index().items():
        summary.append({
            "scope": "error_run_length", "group_value": str(int(length)),
            "total_bits": len(bits), "bit_errors": int(bits.is_error.sum()),
            "error_rate": float(bits.is_error.mean()), "metric": "observed_runs",
            "value": float(count),
        })
    pd.DataFrame(summary).to_csv(args.output_dir / "error_summary.csv", index=False)
    runs.to_csv(args.output_dir / "error_runs.csv", index=False)
    confidence_bins(bits).to_csv(args.output_dir / "correlation_error_bins.csv", index=False)
    carrier_rows(bits).to_csv(args.output_dir / "carrier_loss_analysis.csv", index=False)
    print("decoder error analysis complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source-results", type=Path, default=SOURCE_RESULTS)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--limit-per-partition", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.analyze_only:
        raise ValueError("Ne mogu zajedno --generate-only i --analyze-only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_source.path = args.source_results
    config = analysis_config(args)
    write_or_validate_config(args.output_dir / "config.json", config)
    if not args.analyze_only:
        generate(args, config)
    if not args.generate_only:
        analyze(args)


if __name__ == "__main__":
    main()
