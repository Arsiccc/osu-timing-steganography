"""Development-only physical pilot for an exact eight-coded-bit payload floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.layout_diagnostics import note_quarter_labels
from osu_stego.analysis.timing_features import (
    BASELINE_FEATURES, FEATURE_VERSION, FULL_FEATURES, POSITION_FEATURES,
    residual_timing_features,
)
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import decode_hamming_8_4_secded, encode_hamming_8_4_secded, secded_syndrome
from osu_stego.stego.integrity import corrected_bit_is_not_minimum_decision
from osu_stego.stego.payload_layout import (
    extract_bipolar_message_with_layout, message_correlations_with_layout,
    select_payload_blocks,
)
from scripts.analysis.analyze_capacity_coverage import CAPS, file_sha256
from scripts.experiments.adaptive_layout_strong_common import (
    DEFAULT_PARTITION, LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PN_KEY,
)
from scripts.experiments.run_ecc_equal_payload import array_hash, key_id
from scripts.experiments.run_layout_comparison import build_physical_stego, layout_coverage
from scripts.experiments.run_payload_sweep import deterministic_message, message_length_for_fraction, nominal_capacity_bits
from scripts.experiments.run_pilot_ber_sweep import build_beatmap_index, load_map_offsets, prepare_replay, stable_seed
from scripts.experiments.run_steganalysis import bootstrap_auc_by_group, shuffled_group_ids
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "one-codeword-floor-pilot-v1"
RULE_SHA256 = "f49467d3b69e18b76ba88816d38270fc9c948583166c3a99d9c5ada2bc67ae42"
BIN_EDGES = (-np.inf, .10, .125, .15, .20, np.inf)
BIN_LABELS = ("<=10%", "10-12.5%", "12.5-15%", "15-20%", ">20%")
CAP_POLICIES = (("current_7.5pct", .075), ("floor_cap_10pct", .10),
                ("floor_cap_12.5pct", .125), ("floor_cap_15pct", .15),
                ("floor_cap_20pct", .20), ("floor_uncapped", 1.0))


def selection_hash_value(replay_file: str) -> str:
    return hashlib.sha256(f"{EXPERIMENT_VERSION}|selection|{replay_file}".encode()).hexdigest()


def make_selection(capacity_path: Path, output_path: Path, per_bin: int = 20) -> pd.DataFrame:
    capacity = pd.read_csv(capacity_path)
    candidates = capacity.query(
        "partition == 'development' and eligible_at_7_5pct == 0 and structurally_can_fit_full_codeword == 1"
    ).copy()
    candidates["required_fraction_bin"] = pd.cut(
        candidates.minimum_fraction_1_codewords, BIN_EDGES, labels=BIN_LABELS,
        right=True,
    ).astype(str)
    candidates["selection_hash"] = candidates.replay_file.map(selection_hash_value)
    selected = []
    for label in BIN_LABELS:
        group = candidates[candidates.required_fraction_bin == label].sort_values("selection_hash")
        if len(group):
            selected.append(group.head(per_bin))
    result = pd.concat(selected, ignore_index=True)
    if len(result) != 80:
        raise RuntimeError(f"Očekivano je 80 unapred stratifikovanih replay-a, dobijeno {len(result)}.")
    fields = ["replay_file", "beatmap_hash", "performance_category", "note_index_length",
              "nominal_coded_bit_capacity", "coded_bits_at_7_5pct",
              "minimum_fraction_1_codewords", "required_fraction_bin", "selection_hash"]
    result[fields].to_csv(output_path, index=False)
    return result[fields]


def config_id(replay_file: str, replay_sha: str, seed: int, encoded: np.ndarray) -> str:
    material = "|".join((EXPERIMENT_VERSION, replay_file, replay_sha, str(seed),
                         array_hash(encoded), key_id(PN_KEY), key_id(LAYOUT_KEYS[seed]),
                         str(N_VALUE), "exact-coded-bits=8", RULE_SHA256))
    return hashlib.sha256(material.encode()).hexdigest()


def append_unique(path: Path, rows: list[dict], key: str = "config_id") -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        old = old[~old[key].astype(str).isin(set(incoming[key].astype(str)))]
        incoming = pd.concat([old, incoming], ignore_index=True)
    incoming.sort_values(["replay_file", "layout_seed"]).to_csv(path, index=False)


def generate(args: argparse.Namespace, config: dict) -> None:
    selection = pd.read_csv(args.selection)
    cohort = load_partitioned_cohort(args.results, args.performance_groups, args.partition, "development")
    selected = selection[["replay_file", "required_fraction_bin", "minimum_fraction_1_codewords"]].merge(
        cohort, on="replay_file", validate="one_to_one"
    )
    physical_path = args.output_dir / "physical_results.csv"
    complete_sets: list[set[str]] = []
    for path in (physical_path, args.output_dir / "integrity_results.csv"):
        complete_sets.append(set(pd.read_csv(path).config_id.astype(str)) if path.is_file() else set())
    feature_path = args.output_dir / "timing_features.csv"
    if feature_path.is_file():
        feature_frame = pd.read_csv(feature_path)
        labels = feature_frame.groupby("config_id").label.apply(lambda values: set(values.astype(int)))
        complete_sets.append(set(labels[labels.map(lambda value: value == {0, 1})].index.astype(str)))
    else:
        complete_sets.append(set())
    existing = set.intersection(*complete_sets)
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    policy = load_sender_local_policy(args.policy)
    rule = json.loads(args.frozen_rule.read_text())
    physical_rows: list[dict] = []
    feature_rows: list[dict] = []
    integrity_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="one_codeword_floor_") as temp_name:
        temp_dir = Path(temp_name)
        for replay_index, row in enumerate(selected.itertuples(index=False), 1):
            context = prepare_replay(row, args.dataset_dir, beatmaps, offsets)
            if len(context.original_residuals) != int(row.num_notes):
                raise RuntimeError("Metadata num_notes nije jednak note-indexed residual dužini.")
            capacity = nominal_capacity_bits(len(context.original_residuals), N_VALUE)
            if message_length_for_fraction(capacity, .075) >= 8 or capacity < 8:
                raise RuntimeError("Pilot selection krši floor eligibility definiciju.")
            useful = deterministic_message(context.replay_file, context.beatmap_hash, 0.0,
                                           N_VALUE, .075, 4, MESSAGE_SEED)
            encoded = encode_hamming_8_4_secded(useful)
            if len(encoded) != 8:
                raise RuntimeError("Floor mora imati tačno osam coded bits.")
            alpha = alpha_from_replay(policy, context.replay)
            replay_sha = file_sha256(context.osr_path)
            clean_features = residual_timing_features(context.original_residuals)
            for seed, layout_key in enumerate(LAYOUT_KEYS):
                identity = config_id(context.replay_file, replay_sha, seed, encoded)
                if identity in existing:
                    continue
                _, roundtrip, diagnostics = build_physical_stego(
                    context, encoded, alpha, N_VALUE, "distributed", PN_KEY,
                    layout_key, float(policy["hit_margin_ms"]), temp_dir / f"{identity}.osr",
                )
                received = extract_bipolar_message_with_layout(
                    roundtrip, PN_KEY, layout_key, N_VALUE, 8, "distributed"
                )
                correlations = message_correlations_with_layout(
                    roundtrip, PN_KEY, layout_key, N_VALUE, 8, "distributed"
                )
                decoded, statuses = decode_hamming_8_4_secded(received)
                syndrome, parity = secded_syndrome(received)
                decision = corrected_bit_is_not_minimum_decision(
                    str(statuses[0]), syndrome, parity, np.abs(correlations)
                )
                correct = bool(np.array_equal(decoded, useful))
                outcome = "REJECT" if not decision.accepted else ("CORRECT_ACCEPT" if correct else "WRONG_ACCEPT")
                blocks = select_payload_blocks(len(roundtrip), N_VALUE, 8, layout_key, "distributed")
                selected_positions = np.concatenate([np.arange(block*N_VALUE, (block+1)*N_VALUE) for block in blocks])
                quarter_labels = note_quarter_labels(len(roundtrip))[selected_positions]
                span, occupied_quarters = layout_coverage(len(roundtrip), N_VALUE, 8, "distributed", layout_key)
                common = {
                    "experiment_version": EXPERIMENT_VERSION, "config_id": identity,
                    "replay_file": context.replay_file, "beatmap_hash": context.beatmap_hash,
                    "performance_category": row.performance_category,
                    "required_fraction": float(row.minimum_fraction_1_codewords),
                    "required_fraction_bin": row.required_fraction_bin,
                    "layout_seed": seed, "layout_key_id": key_id(layout_key), "alpha": alpha,
                    "N": N_VALUE, "coded_bits": 8, "useful_bits": 4,
                    "message_sha256": array_hash(useful), "encoded_message_sha256": array_hash(encoded),
                    "replay_sha256": replay_sha, "policy_sha256": config["policy_sha256"],
                    "frozen_rule_sha256": RULE_SHA256,
                }
                requested = int(diagnostics["requested_carriers"])
                active = int(diagnostics["active_carriers"])
                physical_rows.append({
                    **common, "note_index_length": len(roundtrip), "nominal_coded_bit_capacity": capacity,
                    "raw_bit_errors": int(np.sum(received != encoded)),
                    "raw_ber": float(np.mean(received != encoded)),
                    "post_ecc_bit_errors": int(np.sum(decoded != useful)),
                    "post_ecc_ber": float(np.mean(decoded != useful)),
                    "active_carrier_fraction": active/requested if requested else 0.0,
                    "modified_note_fraction": active/len(roundtrip),
                    "selected_note_fraction": 64/len(roundtrip),
                    "selected_timeline_span_fraction": span,
                    "occupied_quarters": occupied_quarters,
                    "energy_per_note_ms2": diagnostics["sum_squared_shift_ms2"]/len(roundtrip),
                    "energy_per_useful_bit_ms2": diagnostics["sum_squared_shift_ms2"]/4,
                    **{f"selected_note_positions_q{q}": int(np.sum(quarter_labels == q)) for q in range(1, 5)},
                    **diagnostics,
                })
                integrity_rows.append({
                    **common, "true_info_bits": json.dumps(useful.tolist()),
                    "encoded_bits": json.dumps(encoded.tolist()), "received_bits": json.dumps(received.tolist()),
                    "decoded_info_bits": json.dumps(decoded.tolist()), "hard_status": str(statuses[0]),
                    "syndrome": syndrome, "overall_parity": parity,
                    "abs_correlations": json.dumps(np.abs(correlations).tolist()),
                    "integrity_accepted": int(decision.accepted), "rejection_reason": decision.reason,
                    "outcome": outcome, "correct_accept": int(outcome == "CORRECT_ACCEPT"),
                    "wrong_accept": int(outcome == "WRONG_ACCEPT"), "reject": int(outcome == "REJECT"),
                    "useful_correctly_accepted_bits": 4 * int(outcome == "CORRECT_ACCEPT"),
                })
                for label, values in ((0, clean_features), (1, residual_timing_features(roundtrip))):
                    feature_rows.append({**common, "label": label, "feature_version": FEATURE_VERSION, **values})
            if replay_index % 10 == 0:
                append_unique(physical_path, physical_rows); physical_rows = []
                append_unique(args.output_dir / "integrity_results.csv", integrity_rows); integrity_rows = []
                append_unique(args.output_dir / "timing_features.csv", feature_rows); feature_rows = []
                print(f"physical pilot {replay_index}/{len(selected)}", flush=True)
    append_unique(physical_path, physical_rows)
    append_unique(args.output_dir / "integrity_results.csv", integrity_rows)
    append_unique(args.output_dir / "timing_features.csv", feature_rows)


def rf_auc(group: pd.DataFrame, names: tuple[str, ...], tag: str, iterations: int) -> tuple[dict, pd.DataFrame]:
    X = group[list(names)].to_numpy(float); y = group.label.to_numpy(np.int8)
    replay = group.replay_file.astype(str).to_numpy()
    groups = shuffled_group_ids(replay, stable_seed(42, EXPERIMENT_VERSION, tag, "folds"))
    split = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    scores = np.full(len(group), np.nan)
    for fold, (train, test) in enumerate(split.split(X, y, groups), 1):
        model = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                       random_state=stable_seed(42, EXPERIMENT_VERSION, tag, fold), n_jobs=1)
        model.fit(X[train], y[train]); scores[test] = model.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, scores))
    low, high = bootstrap_auc_by_group(y, scores, replay, stable_seed(42, EXPERIMENT_VERSION, tag, "bootstrap"), iterations)
    result = {"scope": tag, "feature_set": "", "replay_pairs": group.replay_file.nunique(),
              "rows": len(group), "roc_auc": auc, "auc_ci95_low": low, "auc_ci95_high": high,
              "bootstrap_unit": "replay_file", "bootstrap_iterations": iterations}
    predictions = group[["config_id", "replay_file", "layout_seed", "label", "required_fraction_bin"]].copy()
    predictions["score"] = scores
    return result, predictions


def analyze(args: argparse.Namespace) -> None:
    physical = pd.read_csv(args.output_dir / "physical_results.csv")
    integrity = pd.read_csv(args.output_dir / "integrity_results.csv")
    features = pd.read_csv(args.output_dir / "timing_features.csv")
    expected = len(pd.read_csv(args.selection)) * len(LAYOUT_KEYS)
    if len(physical) != expected or physical.config_id.nunique() != expected:
        raise RuntimeError("Physical pilot nije kompletan.")
    if len(features) != expected * 2 or features.groupby("config_id").label.nunique().min() != 2:
        raise RuntimeError("Feature clean/stego parovi nisu kompletni.")
    auc_rows = []; prediction_frames = []
    feature_sets = {"baseline": tuple(BASELINE_FEATURES), "position_only": tuple(POSITION_FEATURES), "full": tuple(FULL_FEATURES)}
    scopes = [("pooled_five_seeds", features)] + [(f"layout_seed_{seed}", features[features.layout_seed == seed]) for seed in range(5)]
    for scope, frame in scopes:
        for feature_set, names in feature_sets.items():
            row, prediction = rf_auc(frame.reset_index(drop=True), names, f"{scope}:{feature_set}", args.bootstrap_iterations)
            row["feature_set"] = feature_set; auc_rows.append(row)
            prediction["scope"] = scope; prediction["feature_set"] = feature_set; prediction_frames.append(prediction)
    pd.DataFrame(auc_rows).to_csv(args.output_dir / "steganalysis_auc.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    bin_rows = []
    full_pooled = predictions[(predictions.scope == "pooled_five_seeds") & (predictions.feature_set == "full")]
    for label in BIN_LABELS:
        p = physical[physical.required_fraction_bin == label]
        i = integrity[integrity.required_fraction_bin == label]
        pred = full_pooled[full_pooled.required_fraction_bin == label]
        row = {"required_fraction_bin": label, "replays": p.replay_file.nunique(), "configs": len(p),
               "raw_ber": p.raw_bit_errors.sum()/(8*len(p)) if len(p) else np.nan,
               "correct_accept_rate": i.correct_accept.mean() if len(i) else np.nan,
               "wrong_accept_rate": i.wrong_accept.mean() if len(i) else np.nan,
               "reject_rate": i.reject.mean() if len(i) else np.nan,
               "mean_modified_note_fraction": p.modified_note_fraction.mean() if len(p) else np.nan,
               "mean_active_carrier_fraction": p.active_carrier_fraction.mean() if len(p) else np.nan,
               "mean_full_clean_score": pred.loc[pred.label == 0, "score"].mean() if len(pred) else np.nan,
               "mean_full_stego_score": pred.loc[pred.label == 1, "score"].mean() if len(pred) else np.nan,
               "auc_reported": int(p.replay_file.nunique() >= 20)}
        row["full_auc_descriptive"] = float(roc_auc_score(pred.label, pred.score)) if row["auc_reported"] else np.nan
        bin_rows.append(row)
    pd.DataFrame(bin_rows).to_csv(args.output_dir / "payload_fraction_bins.csv", index=False)

    metric_fields = ["requested_carriers", "active_carriers", "active_carrier_fraction", "dropped_for_hit_window",
                     "dropped_for_chronology", "new_unmatched_events", "changed_match_status_total",
                     "sum_squared_shift_ms2", "rms_applied_shift_ms", "mean_absolute_applied_shift_ms",
                     "modified_note_fraction", "selected_note_fraction", "selected_timeline_span_fraction",
                     "energy_per_note_ms2", "energy_per_useful_bit_ms2"]
    cost = physical.groupby("required_fraction_bin", observed=False)[metric_fields].mean().reset_index()
    cost.to_csv(args.output_dir / "physical_cost.csv", index=False)

    diagnostics = [{"check": "physical_configs", "value": len(physical), "expected": expected, "passed": int(len(physical)==expected)},
                   {"check": "coded_bits_exactly_8", "value": int(physical.coded_bits.nunique()==1 and physical.coded_bits.iloc[0]==8), "expected": 1, "passed": int(physical.coded_bits.nunique()==1 and physical.coded_bits.iloc[0]==8)},
                   {"check": "useful_bits_exactly_4", "value": int((physical.useful_bits==4).all()), "expected": 1, "passed": int((physical.useful_bits==4).all())},
                   {"check": "integrity_rule_hash", "value": str(physical.frozen_rule_sha256.iloc[0]), "expected": RULE_SHA256, "passed": int((physical.frozen_rule_sha256==RULE_SHA256).all())},
                   {"check": "duplicate_config_ids", "value": int(physical.config_id.duplicated().sum()), "expected": 0, "passed": int(not physical.config_id.duplicated().any())}]
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "one_codeword_floor_pilot_v1")
    parser.add_argument("--capacity", type=Path, default=RESULTS_DIR / "capacity_coverage_v1" / "capacity_by_replay.csv")
    parser.add_argument("--selection", type=Path, default=RESULTS_DIR / "one_codeword_floor_pilot_v1" / "pilot_selection.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--policy", type=Path, default=CONFIG_DIR / "adaptive_alpha_sender_local_v2.json")
    parser.add_argument("--frozen-rule", type=Path, default=CONFIG_DIR / "ecc_integrity_rejection_v1.json")
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.frozen_rule) != RULE_SHA256:
        raise RuntimeError("Frozen integrity rule hash je promenjen.")
    if not args.selection.is_file():
        make_selection(args.capacity, args.selection)
    config = {"experiment_version": EXPERIMENT_VERSION, "scientific_status": "development_exploratory",
              "selection_sha256": file_sha256(args.selection), "capacity_audit_sha256": file_sha256(args.capacity),
              "policy_sha256": file_sha256(args.policy), "frozen_rule_sha256": file_sha256(args.frozen_rule),
              "partition_sha256": file_sha256(args.partition), "map_offsets_sha256": file_sha256(args.map_offsets),
              "cohort_sha256": file_sha256(args.results), "performance_groups_sha256": file_sha256(args.performance_groups),
              "N": N_VALUE, "exact_coded_bits": 8, "useful_bits": 4, "layout": "distributed",
              "layout_key_ids": [key_id(value) for value in LAYOUT_KEYS], "pn_key_id": key_id(PN_KEY),
              "message_seed": MESSAGE_SEED, "candidate_caps": [name for name, _ in CAP_POLICIES],
              "rf": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2, "n_jobs": 1, "folds": 5},
              "bootstrap_iterations": args.bootstrap_iterations}
    config["source_sha256"] = {
        str(path): file_sha256(path) for path in (
            Path(__file__), Path("osu_stego/stego/payload_layout.py"),
            Path("osu_stego/stego/ecc.py"), Path("osu_stego/stego/integrity.py"),
            Path("osu_stego/stego/adaptive_alpha.py"),
            Path("osu_stego/analysis/timing_features.py"),
            Path("scripts/experiments/run_layout_comparison.py"),
            Path("scripts/experiments/run_pilot_ber_sweep.py"),
        )
    }
    config_path = args.output_dir / "config.json"
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if (config_path.is_file() and config_path.read_text() != serialized
            and (args.output_dir / "physical_results.csv").is_file()):
        raise RuntimeError("Postojeći pilot config se razlikuje nakon početka physical run-a.")
    config_path.write_text(serialized)
    if args.select_only:
        print(f"selection={args.selection} sha256={file_sha256(args.selection)}"); return
    if not args.analyze_only:
        generate(args, config)
    analyze(args)


if __name__ == "__main__":
    main()
