"""One-shot new-map validation of the frozen 12.5%-capped payload floor."""

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

from osu_stego.analysis.timing_features import BASELINE_FEATURES, FEATURE_VERSION, FULL_FEATURES, POSITION_FEATURES, residual_timing_features
from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import decode_hamming_8_4_secded, encode_hamming_8_4_secded, secded_syndrome
from osu_stego.stego.integrity import baseline_integrity_decision, corrected_bit_is_not_minimum_decision
from osu_stego.stego.payload_layout import extract_bipolar_message_with_layout, message_correlations_with_layout
from osu_stego.stego.payload_policy import one_codeword_floor_decision
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PN_KEY
from scripts.experiments.run_adaptive_layout_strong import LEGACY_STRONG_EXPERIMENT
from scripts.experiments.run_ecc_equal_payload import array_hash, key_id
from scripts.experiments.run_layout_comparison import build_physical_stego, layout_coverage
from scripts.experiments.run_payload_sweep import deterministic_message
from scripts.experiments.run_pilot_ber_sweep import build_beatmap_index, load_map_offsets, prepare_replay, stable_seed
from scripts.experiments.run_steganalysis import bootstrap_auc_by_group


EXPERIMENT_VERSION = "one-codeword-floor-validation-v1"
EXPECTED_POLICY_SHA256 = "92d9224974838a0628aee9df0c3d112b6c86fb981fe057a3c31acb2daf22ac78"
EXPECTED_INTEGRITY_SHA256 = "f49467d3b69e18b76ba88816d38270fc9c948583166c3a99d9c5ada2bc67ae42"
EXPECTED_ADAPTIVE_SHA256 = "ce689569361c4691e68d6be22bb400acbf881b70de78eea123af948cd0f69d9a"
FEATURE_SETS = {"baseline": tuple(BASELINE_FEATURES), "position_only": tuple(POSITION_FEATURES), "full": tuple(FULL_FEATURES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, pd.DataFrame]:
    lock = json.loads(args.validation_lock.read_text(encoding="utf-8"))
    if sha256_file(args.payload_policy) != EXPECTED_POLICY_SHA256 or lock["payload_policy_sha256"] != EXPECTED_POLICY_SHA256:
        raise RuntimeError("Frozen payload policy hash mismatch.")
    if sha256_file(args.integrity_rule) != EXPECTED_INTEGRITY_SHA256 or lock["integrity_rule_sha256"] != EXPECTED_INTEGRITY_SHA256:
        raise RuntimeError("Frozen integrity hash mismatch.")
    if sha256_file(args.adaptive_policy) != EXPECTED_ADAPTIVE_SHA256 or lock["adaptive_alpha_policy_sha256"] != EXPECTED_ADAPTIVE_SHA256:
        raise RuntimeError("Frozen adaptive-alpha hash mismatch.")
    if sha256_file(args.manifest) != lock["corpus_manifest_sha256"] or sha256_file(args.map_offsets) != lock["map_offsets_sha256"]:
        raise RuntimeError("Corpus manifest ili map offsets nisu locked input.")
    if sha256_file(Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("Validation runner se promenio posle lock-a.")
    return (json.loads(args.payload_policy.read_text()), json.loads(args.integrity_rule.read_text()),
            load_sender_local_policy(args.adaptive_policy), pd.read_csv(args.manifest))


def capacity_classification(policy: dict, manifest: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    beatmaps = build_beatmap_index(dataset_dir)
    rows = []
    for source in manifest.itertuples(index=False):
        note_count = len(beatmaps[str(source.beatmap_hash)].note_times)
        decision = one_codeword_floor_decision(policy, note_count)
        rows.append({
            "replay_file": source.replay_file, "replay_hash": source.replay_hash,
            "beatmap_hash": source.beatmap_hash, "username": source.username,
            "note_index_length": note_count, "nominal_capacity_bits": decision.nominal_capacity_bits,
            "normal_allocated_bits": decision.normal_allocated_bits,
            "capacity_class": decision.capacity_class,
            "transmitted_coded_bits": decision.transmitted_coded_bits,
            "useful_bits": decision.useful_bits,
            "required_floor_fraction": (8 / decision.nominal_capacity_bits if decision.capacity_class == "FLOOR_12_5" else np.nan),
        })
    return pd.DataFrame(rows)


def physical_identifier(replay_file: str, replay_sha: str, seed: int, encoded: np.ndarray, capacity_class: str) -> str:
    material = "|".join((EXPERIMENT_VERSION, replay_file, replay_sha, capacity_class, str(seed),
                         array_hash(encoded), key_id(PN_KEY), key_id(LAYOUT_KEYS[seed]),
                         EXPECTED_POLICY_SHA256, EXPECTED_ADAPTIVE_SHA256, EXPECTED_INTEGRITY_SHA256))
    return hashlib.sha256(material.encode()).hexdigest()


def append_unique(path: Path, rows: list[dict], key_columns: tuple[str, ...]) -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        incoming_keys = set(map(tuple, incoming[list(key_columns)].astype(str).to_numpy()))
        old_keys = old[list(key_columns)].astype(str).apply(tuple, axis=1)
        old = old[~old_keys.isin(incoming_keys)]
        incoming = pd.concat([old, incoming], ignore_index=True)
    incoming.to_csv(path, index=False)


def completed_ids(output_dir: Path) -> set[str]:
    required = (("physical_results.csv", None), ("message_results.csv", 2),
                ("codeword_results.csv", None), ("timing_features.csv", 2))
    sets = []
    for filename, expected_count in required:
        path = output_dir / filename
        if not path.is_file():
            sets.append(set()); continue
        frame = pd.read_csv(path)
        if expected_count is None:
            sets.append(set(frame.config_id.astype(str)))
        else:
            counts = frame.groupby("config_id").size()
            sets.append(set(counts[counts == expected_count].index.astype(str)))
    return set.intersection(*sets)


def message_outcome(accepted: bool, correct: bool) -> str:
    return "REJECT" if not accepted else ("CORRECT_ACCEPT" if correct else "WRONG_ACCEPT")


def generate(args: argparse.Namespace, policy: dict, integrity_rule: dict, adaptive_policy: dict,
             manifest: pd.DataFrame, classification: pd.DataFrame) -> int:
    selected = classification.merge(manifest[["replay_file", "source_replay_sha256"]], on="replay_file", validate="one_to_one")
    selected = selected[selected.capacity_class != "NO_CAPACITY"].copy()
    if args.limit_replays is not None:
        selected = selected.sort_values(["beatmap_hash", "replay_file"]).head(args.limit_replays)
    complete = completed_ids(args.output_dir)
    beatmaps = build_beatmap_index(args.dataset_dir); offsets = load_map_offsets(args.map_offsets)
    physical_rows: list[dict] = []; message_rows: list[dict] = []; word_rows: list[dict] = []; feature_rows: list[dict] = []
    new_configs = 0
    with tempfile.TemporaryDirectory(prefix="floor_validation_") as temp_name:
        for replay_index, source in enumerate(selected.itertuples(index=False), 1):
            row = pd.Series({"skill_category": "New", "replay_file": source.replay_file,
                             "beatmap_hash": source.beatmap_hash})
            context = prepare_replay(row, args.dataset_dir, beatmaps, offsets)
            if len(context.original_residuals) != int(source.note_index_length):
                raise RuntimeError("Primary note-index length mismatch.")
            alpha = alpha_from_replay(adaptive_policy, context.replay)
            useful = deterministic_message(context.replay_file, context.beatmap_hash, 0.0, N_VALUE, .075,
                                           int(source.useful_bits), MESSAGE_SEED)
            encoded = encode_hamming_8_4_secded(useful)
            if len(encoded) != int(source.transmitted_coded_bits):
                raise RuntimeError("Payload policy i SECDED encoder se ne slažu.")
            replay_sha = sha256_file(context.osr_path)
            if replay_sha != source.source_replay_sha256:
                raise RuntimeError("Primary replay hash se promenio nakon manifest lock-a.")
            clean_features = residual_timing_features(context.original_residuals)
            for seed, layout_key in enumerate(LAYOUT_KEYS):
                identity = physical_identifier(context.replay_file, replay_sha, seed, encoded, source.capacity_class)
                if identity in complete:
                    continue
                _, roundtrip, diagnostics = build_physical_stego(
                    context, encoded, alpha, N_VALUE, "distributed", PN_KEY, layout_key,
                    float(adaptive_policy["hit_margin_ms"]), Path(temp_name) / f"{identity}.osr",
                )
                received = extract_bipolar_message_with_layout(roundtrip, PN_KEY, layout_key, N_VALUE, len(encoded), "distributed")
                correlations = message_correlations_with_layout(roundtrip, PN_KEY, layout_key, N_VALUE, len(encoded), "distributed")
                decoded, statuses = decode_hamming_8_4_secded(received)
                word_count = len(encoded) // 8
                common = {
                    "experiment_version": EXPERIMENT_VERSION, "config_id": identity,
                    "replay_file": context.replay_file, "beatmap_hash": context.beatmap_hash,
                    "username": source.username, "capacity_class": source.capacity_class,
                    "layout_seed": seed, "layout_key_id": key_id(layout_key), "alpha": alpha, "N": N_VALUE,
                    "nominal_capacity_bits": source.nominal_capacity_bits,
                    "normal_allocated_bits": source.normal_allocated_bits,
                    "coded_bits": len(encoded), "useful_bits": len(useful),
                    "message_sha256": array_hash(useful), "encoded_message_sha256": array_hash(encoded),
                    "replay_sha256": replay_sha, "payload_policy_sha256": EXPECTED_POLICY_SHA256,
                    "adaptive_policy_sha256": EXPECTED_ADAPTIVE_SHA256,
                    "integrity_rule_sha256": EXPECTED_INTEGRITY_SHA256,
                }
                baseline_accepts = []; integrity_accepts = []; word_correct = []
                for word_index in range(word_count):
                    code_slice = slice(word_index*8, (word_index+1)*8); info_slice = slice(word_index*4, (word_index+1)*4)
                    syndrome, parity = secded_syndrome(received[code_slice])
                    baseline = baseline_integrity_decision(str(statuses[word_index]))
                    integrity = corrected_bit_is_not_minimum_decision(
                        str(statuses[word_index]), syndrome, parity, np.abs(correlations[code_slice]))
                    correct = bool(np.array_equal(decoded[info_slice], useful[info_slice]))
                    baseline_accepts.append(baseline.accepted); integrity_accepts.append(integrity.accepted); word_correct.append(correct)
                    rank = np.nan
                    if str(statuses[word_index]) == "corrected_single":
                        position = syndrome - 1 if syndrome else 7
                        confidence = np.abs(correlations[code_slice])
                        rank = int(np.argsort(np.argsort(confidence, kind="stable"), kind="stable")[position])
                    word_rows.append({
                        **common, "codeword_index": word_index, "hard_status": str(statuses[word_index]),
                        "syndrome": syndrome, "overall_parity": parity,
                        "raw_error_count": int(np.sum(received[code_slice] != encoded[code_slice])),
                        "word_ground_truth_correct": int(correct), "baseline_accepted": int(baseline.accepted),
                        "baseline_reason": baseline.reason, "integrity_accepted": int(integrity.accepted),
                        "integrity_reason": integrity.reason, "corrected_bit_confidence_rank": rank,
                        "abs_correlations": json.dumps(np.abs(correlations[code_slice]).tolist()),
                    })
                all_correct = all(word_correct)
                for method, accepts in (("baseline", baseline_accepts), ("integrity", integrity_accepts)):
                    accepted = all(accepts); outcome = message_outcome(accepted, all_correct)
                    message_rows.append({
                        **common, "method": method, "outcome": outcome,
                        "correct_accept": int(outcome == "CORRECT_ACCEPT"),
                        "wrong_accept": int(outcome == "WRONG_ACCEPT"), "reject": int(outcome == "REJECT"),
                        "accepted": int(accepted), "ground_truth_full_message_correct": int(all_correct),
                        "accepted_message_correct": int(accepted and all_correct),
                        "useful_correctly_accepted_bits": len(useful) * int(outcome == "CORRECT_ACCEPT"),
                    })
                requested = int(diagnostics["requested_carriers"]); active = int(diagnostics["active_carriers"])
                span, quarters = layout_coverage(len(roundtrip), N_VALUE, len(encoded), "distributed", layout_key)
                physical_rows.append({
                    **common, "note_index_length": len(roundtrip),
                    "raw_bit_errors": int(np.sum(received != encoded)), "raw_ber": float(np.mean(received != encoded)),
                    "post_ecc_bit_errors": int(np.sum(decoded != useful)), "post_ecc_ber": float(np.mean(decoded != useful)),
                    "codewords": word_count, "active_carrier_fraction": active/requested if requested else 0.0,
                    "modified_note_fraction": active/len(roundtrip), "selected_note_fraction": len(encoded)*N_VALUE/len(roundtrip),
                    "block_span_fraction": span, "occupied_quarters": quarters,
                    "energy_per_note_ms2": diagnostics["sum_squared_shift_ms2"]/len(roundtrip),
                    "energy_per_useful_bit_ms2": diagnostics["sum_squared_shift_ms2"]/len(useful), **diagnostics,
                })
                for label, values in ((0, clean_features), (1, residual_timing_features(roundtrip))):
                    feature_rows.append({**common, "label": label, "feature_version": FEATURE_VERSION, **values})
                new_configs += 1
            if replay_index % 10 == 0:
                append_unique(args.output_dir/"physical_results.csv", physical_rows, ("config_id",)); physical_rows=[]
                append_unique(args.output_dir/"message_results.csv", message_rows, ("config_id","method")); message_rows=[]
                append_unique(args.output_dir/"codeword_results.csv", word_rows, ("config_id","codeword_index")); word_rows=[]
                append_unique(args.output_dir/"timing_features.csv", feature_rows, ("config_id","label")); feature_rows=[]
                print(f"physical {replay_index}/{len(selected)} new_configs={new_configs}", flush=True)
    append_unique(args.output_dir/"physical_results.csv", physical_rows, ("config_id",))
    append_unique(args.output_dir/"message_results.csv", message_rows, ("config_id","method"))
    append_unique(args.output_dir/"codeword_results.csv", word_rows, ("config_id","codeword_index"))
    append_unique(args.output_dir/"timing_features.csv", feature_rows, ("config_id","label"))
    return new_configs


def detector_generalization(new_features: pd.DataFrame, old_features: pd.DataFrame, iterations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = old_features[(old_features.partition == "development") &
                       (old_features.alpha_method == "sender_local_adaptive") &
                       (old_features.layout == "distributed")].copy()
    if set(old.beatmap_hash.astype(str)) & set(new_features.beatmap_hash.astype(str)):
        raise RuntimeError("Detector train/test beatmap overlap.")
    rows = []; predictions = []
    for seed in range(5):
        train = old[old.layout_seed.astype(str) == str(seed)]
        test_seed = new_features[new_features.layout_seed.astype(int) == seed]
        for feature_set, names in FEATURE_SETS.items():
            classifier = RandomForestClassifier(
                n_estimators=300, max_features="sqrt", min_samples_leaf=2, n_jobs=1,
                random_state=stable_seed(MESSAGE_SEED, LEGACY_STRONG_EXPERIMENT, "generalization-rf"),
            )
            classifier.fit(train[list(names)].to_numpy(float), train.label.to_numpy(np.int8))
            scores = classifier.predict_proba(test_seed[list(names)].to_numpy(float))[:, 1]
            scored = test_seed[["config_id","replay_file","beatmap_hash","capacity_class","layout_seed","label"]].copy()
            scored["feature_set"] = feature_set; scored["score"] = scores; predictions.append(scored)
            for capacity_class, subset in [("ALL_ATTEMPTED", scored), *list(scored.groupby("capacity_class"))]:
                if subset.label.nunique() != 2:
                    continue
                auc = float(roc_auc_score(subset.label, subset.score))
                low, high = bootstrap_auc_by_group(
                    subset.label.to_numpy(np.int8), subset.score.to_numpy(float), subset.beatmap_hash.astype(str).to_numpy(),
                    stable_seed(42, EXPERIMENT_VERSION, seed, feature_set, capacity_class, "map-bootstrap"), iterations,
                )
                rows.append({"scope":"frozen_development_to_new_maps", "capacity_class":capacity_class,
                             "layout_seed":seed, "feature_set":feature_set, "replay_pairs":subset.replay_file.nunique(),
                             "beatmaps":subset.beatmap_hash.nunique(), "roc_auc":auc, "auc_ci95_low":low,
                             "auc_ci95_high":high, "bootstrap_unit":"beatmap_hash", "bootstrap_iterations":iterations,
                             "train_replay_pairs":train.replay_file.nunique(), "trees":300})
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def summarize(args: argparse.Namespace, classification: pd.DataFrame, new_configs: int) -> None:
    physical = pd.read_csv(args.output_dir/"physical_results.csv"); messages = pd.read_csv(args.output_dir/"message_results.csv")
    words = pd.read_csv(args.output_dir/"codeword_results.csv"); features = pd.read_csv(args.output_dir/"timing_features.csv")
    attempted = classification[classification.capacity_class != "NO_CAPACITY"]
    expected = len(attempted)*5 if args.limit_replays is None else args.limit_replays*5
    if len(physical) != expected or physical.config_id.nunique() != expected:
        raise RuntimeError("Physical output nije kompletan/jedinstven.")
    integrity = messages[messages.method == "integrity"]
    reliability = []
    for capacity_class, group in [("ALL_ATTEMPTED", physical), *list(physical.groupby("capacity_class"))]:
        msg = integrity[integrity.config_id.isin(group.config_id)]
        reliability.append({"capacity_class":capacity_class, "configs":len(group), "replays":group.replay_file.nunique(),
                            "raw_bit_errors":int(group.raw_bit_errors.sum()), "coded_bits":int(group.coded_bits.sum()),
                            "raw_ber":group.raw_bit_errors.sum()/group.coded_bits.sum(),
                            "post_ecc_bit_errors":int(group.post_ecc_bit_errors.sum()), "useful_bits":int(group.useful_bits.sum()),
                            "post_ecc_ber":group.post_ecc_bit_errors.sum()/group.useful_bits.sum(),
                            "correct_accept":int(msg.correct_accept.sum()), "wrong_accept":int(msg.wrong_accept.sum()),
                            "reject":int(msg.reject.sum()), "correct_accept_rate":msg.correct_accept.mean(),
                            "wrong_accept_rate":msg.wrong_accept.mean(), "reject_rate":msg.reject.mean(),
                            "accepted_message_accuracy":msg.correct_accept.sum()/msg.accepted.sum() if msg.accepted.sum() else np.nan,
                            "mean_useful_bits_per_config":group.useful_bits.mean(),
                            "correctly_accepted_useful_bits_per_config":msg.useful_correctly_accepted_bits.mean()})
    pd.DataFrame(reliability).to_csv(args.output_dir/"reliability_by_class.csv",index=False)
    counts = classification.capacity_class.value_counts()
    total = len(classification); system = []
    for seed in range(5):
        msg = integrity[integrity.layout_seed == seed]
        system.append({"layout_seed":seed,"input_replays":total,"normal_coverage":counts.get("NORMAL_7_5",0)/total,
                       "floor_coverage_gain":counts.get("FLOOR_12_5",0)/total,"total_attempt_coverage":len(attempted)/total,
                       "no_capacity_abstention":counts.get("NO_CAPACITY",0)/total,
                       "correct_delivery":msg.correct_accept.sum()/total,"silent_wrong_delivery":msg.wrong_accept.sum()/total,
                       "explicit_decoder_rejection":msg.reject.sum()/total,
                       "correctly_accepted_useful_information_per_input_replay":msg.useful_correctly_accepted_bits.sum()/total})
    pd.DataFrame(system).to_csv(args.output_dir/"system_metrics.csv",index=False)
    physical.groupby("capacity_class")[["requested_carriers","active_carriers","active_carrier_fraction","dropped_for_hit_window",
        "dropped_for_chronology","new_unmatched_events","new_matched_events","changed_match_status_total","sum_squared_shift_ms2",
        "rms_applied_shift_ms","mean_absolute_applied_shift_ms","modified_note_fraction","energy_per_note_ms2","energy_per_useful_bit_ms2"]].mean().reset_index().to_csv(args.output_dir/"physical_by_class.csv",index=False)
    by_seed = physical.groupby(["capacity_class","layout_seed"]).agg(configs=("config_id","size"),replays=("replay_file","nunique"),
        raw_errors=("raw_bit_errors","sum"),coded_bits=("coded_bits","sum"),active_fraction=("active_carrier_fraction","mean"),
        new_unmatched=("new_unmatched_events","mean"),rematching_changes=("changed_match_status_total","mean")).reset_index()
    by_seed["raw_ber"] = by_seed.raw_errors/by_seed.coded_bits
    outcomes = integrity.groupby(["capacity_class","layout_seed"])[["correct_accept","wrong_accept","reject"]].mean().reset_index()
    by_seed.merge(outcomes,on=["capacity_class","layout_seed"]).to_csv(args.output_dir/"results_by_seed.csv",index=False)
    by_map = physical.groupby(["beatmap_hash","capacity_class"]).agg(configs=("config_id","size"),replays=("replay_file","nunique"),raw_errors=("raw_bit_errors","sum"),coded_bits=("coded_bits","sum")).reset_index()
    by_map["raw_ber"] = by_map.raw_errors/by_map.coded_bits
    map_outcomes=integrity.groupby(["beatmap_hash","capacity_class"])[["correct_accept","wrong_accept","reject"]].mean().reset_index()
    by_map.merge(map_outcomes,on=["beatmap_hash","capacity_class"],how="left").to_csv(args.output_dir/"results_by_map.csv",index=False)
    detector, predictions = detector_generalization(features, pd.read_csv(RESULTS_DIR/"adaptive_layout_strong_v1"/"features.csv"), args.bootstrap_iterations)
    detector.to_csv(args.output_dir/"detector_auc.csv",index=False)
    detector.to_csv(args.output_dir/"detector_by_seed.csv",index=False)
    map_auc=[]
    full=predictions[predictions.feature_set=="full"]
    for (beatmap,capacity_class,seed), group in full.groupby(["beatmap_hash","capacity_class","layout_seed"]):
        if group.replay_file.nunique()>=10 and group.label.nunique()==2:
            map_auc.append({"beatmap_hash":beatmap,"capacity_class":capacity_class,"layout_seed":seed,
                            "replay_pairs":group.replay_file.nunique(),"full_auc":roc_auc_score(group.label,group.score)})
    pd.DataFrame(map_auc).to_csv(args.output_dir/"detector_by_map.csv",index=False)
    floor_scores=full[full.capacity_class=="FLOOR_12_5"].merge(
        classification[["replay_file","required_floor_fraction"]],on="replay_file",validate="many_to_one").merge(
        physical[["config_id","modified_note_fraction","energy_per_note_ms2"]],on="config_id",validate="many_to_one")
    stego=floor_scores[floor_scores.label==1]
    pd.DataFrame([{"predictor":name,"spearman_with_full_score":stego[name].rank().corr(stego.score.rank()),
                   "configs":len(stego),"replays":stego.replay_file.nunique()} for name in
                  ("required_floor_fraction","modified_note_fraction","energy_per_note_ms2")]).to_csv(args.output_dir/"floor_detectability.csv",index=False)
    integrity_rows=[]
    for capacity_class, group in words.groupby("capacity_class"):
        baseline_wrong=(~group.word_ground_truth_correct.astype(bool)) & group.baseline_accepted.astype(bool)
        caught=baseline_wrong & ~group.integrity_accepted.astype(bool)
        baseline_correct=group.word_ground_truth_correct.astype(bool) & group.baseline_accepted.astype(bool)
        false_reject=baseline_correct & ~group.integrity_accepted.astype(bool)
        integrity_rows.append({"capacity_class":capacity_class,"codewords":len(group),"baseline_wrong_accept_words":int(baseline_wrong.sum()),
                               "silent_captured":int(caught.sum()),"silent_capture_rate":caught.sum()/baseline_wrong.sum() if baseline_wrong.any() else np.nan,
                               "baseline_correct_accept_words":int(baseline_correct.sum()),"additional_false_rejects":int(false_reject.sum()),
                               "false_rejection_rate":false_reject.sum()/baseline_correct.sum() if baseline_correct.any() else np.nan,
                               "corrected_rank_mean":group.corrected_bit_confidence_rank.mean()})
    pd.DataFrame(integrity_rows).to_csv(args.output_dir/"integrity_by_class.csv",index=False)
    pd.DataFrame([{"check":"new_configs_this_run","value":new_configs},{"check":"physical_configs","value":len(physical)},
                  {"check":"expected_configs","value":expected},{"check":"duplicate_physical","value":physical.config_id.duplicated().sum()},
                  {"check":"historical_map_overlap","value":0},{"check":"payload_policy_sha256","value":EXPECTED_POLICY_SHA256},
                  {"check":"integrity_sha256","value":EXPECTED_INTEGRITY_SHA256},{"check":"adaptive_sha256","value":EXPECTED_ADAPTIVE_SHA256}]).to_csv(args.output_dir/"diagnostics.csv",index=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=RESULTS_DIR/"one_codeword_floor_validation_v1")
    parser.add_argument("--dataset-dir",type=Path,default=Path("data/one_codeword_floor_validation_v1/primary"))
    parser.add_argument("--manifest",type=Path,default=RESULTS_DIR/"one_codeword_floor_validation_v1"/"corpus_manifest.csv")
    parser.add_argument("--validation-lock",type=Path,default=RESULTS_DIR/"one_codeword_floor_validation_v1"/"validation_lock.json")
    parser.add_argument("--payload-policy",type=Path,default=CONFIG_DIR/"payload_policy_one_codeword_floor_v1.json")
    parser.add_argument("--adaptive-policy",type=Path,default=CONFIG_DIR/"adaptive_alpha_sender_local_v2.json")
    parser.add_argument("--integrity-rule",type=Path,default=CONFIG_DIR/"ecc_integrity_rejection_v1.json")
    parser.add_argument("--map-offsets",type=Path,default=CONFIG_DIR/"map_time_offsets_floor_validation_v1.json")
    parser.add_argument("--bootstrap-iterations",type=int,default=2000)
    parser.add_argument("--classify-only",action="store_true"); parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--limit-replays",type=int,default=None)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    policy,integrity,adaptive,manifest=load_locked_inputs(args)
    classification=capacity_classification(policy,manifest,args.dataset_dir)
    classification.to_csv(args.output_dir/"capacity_classification.csv",index=False)
    if args.classify_only:
        print(classification.capacity_class.value_counts().to_string()); return
    new_configs=0 if args.evaluate_only else generate(args,policy,integrity,adaptive,manifest,classification)
    summarize(args,classification,new_configs)
    print(f"validation complete new_configs={new_configs}")


if __name__=="__main__":
    main()
