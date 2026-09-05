"""Frozen physical pilot for robustness across preregistered PN keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.analysis.decoder_errors import bit_level_diagnostics
from osu_stego.analysis.timing_features import FEATURE_VERSION, residual_timing_features
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import decode_hamming_8_4_secded, encode_hamming_8_4_secded, secded_syndrome
from osu_stego.stego.integrity import corrected_bit_is_not_minimum_decision
from osu_stego.stego.payload_layout import extract_bipolar_message_with_layout, message_correlations_with_layout
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PAYLOAD_FRACTION
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import array_hash
from scripts.experiments.run_layout_comparison import build_physical_stego, key_id
from scripts.experiments.run_payload_sweep import deterministic_message
from scripts.experiments.run_pilot_ber_sweep import build_beatmap_index, load_map_offsets, prepare_replay


VERSION = "pn-key-robustness-v1"
OUTPUT = RESULTS_DIR / "pn_key_robustness_v1"


def append_unique(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new_ids = set(map(tuple, incoming[list(keys)].astype(str).to_numpy()))
        old_ids = old[list(keys)].astype(str).apply(tuple, axis=1)
        old = old[~old_ids.isin(new_ids)]
        incoming = pd.concat([old, incoming], ignore_index=True)
    incoming.to_csv(path, index=False)


def config_id(replay_file: str, replay_sha: str, pn_id: str, layout_seed: int, encoded: np.ndarray) -> str:
    material = "|".join((VERSION, replay_file, replay_sha, pn_id, str(layout_seed), array_hash(encoded)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def completed(output: Path) -> set[str]:
    specs = (("physical_results.csv", 1), ("message_results.csv", 1), ("timing_features.csv", 2))
    found = []
    for name, count in specs:
        path = output / name
        if not path.exists():
            found.append(set()); continue
        frame = pd.read_csv(path)
        counts = frame.groupby("config_id").size()
        found.append(set(counts[counts == count].index.astype(str)))
    return set.intersection(*found)


def load_locked(args: argparse.Namespace) -> tuple[dict, dict, pd.DataFrame]:
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    for path, field in ((args.pn_config, "pn_config_sha256"), (args.selection, "pilot_selection_sha256"),
                        (args.adaptive_policy, "adaptive_policy_sha256"), (args.integrity_rule, "integrity_rule_sha256")):
        if file_sha256(path) != lock[field]:
            raise RuntimeError(f"Locked hash mismatch: {path}")
    if file_sha256(Path(__file__)) != lock["source_sha256"]["runner"]:
        raise RuntimeError("Runner je promenjen nakon lock-a.")
    return json.loads(args.pn_config.read_text()), load_sender_local_policy(args.adaptive_policy), pd.read_csv(args.selection)


def generate(args: argparse.Namespace, pn_config: dict, adaptive: dict, selection: pd.DataFrame) -> int:
    selected = selection[selection.eligible_7_5 == 1].sort_values("selection_order")
    if args.limit_replays is not None:
        selected = selected.head(args.limit_replays)
    keys = [pn_config["historical_reference"], *pn_config["new_keys"]]
    complete = completed(args.output_dir)
    beatmaps, offsets = build_beatmap_index(args.dataset_dir), load_map_offsets(args.map_offsets)
    new = 0
    buffers = {name: [] for name in ("physical", "message", "codeword", "bit", "features", "wrong")}
    with tempfile.TemporaryDirectory(prefix="pn_robustness_") as temp:
        for replay_number, source in enumerate(selected.itertuples(index=False), 1):
            context = prepare_replay(source, args.dataset_dir, beatmaps, offsets)
            alpha = alpha_from_replay(adaptive, context.replay)
            useful = deterministic_message(context.replay_file, context.beatmap_hash, 0, N_VALUE, PAYLOAD_FRACTION,
                                           int(source.useful_bits), MESSAGE_SEED)
            encoded = encode_hamming_8_4_secded(useful)
            replay_sha = file_sha256(context.osr_path)
            clean_features = residual_timing_features(context.original_residuals)
            for pn_index, pn in enumerate(keys):
                pn_key, pn_id = pn["key"], pn["key_id"]
                wrong_key = keys[(pn_index + 1) % len(keys)]
                for seed, layout_key in enumerate(LAYOUT_KEYS):
                    identity = config_id(context.replay_file, replay_sha, pn_id, seed, encoded)
                    if identity in complete:
                        continue
                    _, roundtrip, diagnostics = build_physical_stego(
                        context, encoded, alpha, N_VALUE, "distributed", pn_key, layout_key,
                        float(adaptive["hit_margin_ms"]), Path(temp) / f"{identity}.osr")
                    received = extract_bipolar_message_with_layout(roundtrip, pn_key, layout_key, N_VALUE, len(encoded), "distributed")
                    corr = message_correlations_with_layout(roundtrip, pn_key, layout_key, N_VALUE, len(encoded), "distributed")
                    decoded, statuses = decode_hamming_8_4_secded(received)
                    decisions, words_correct = [], []
                    common = {"experiment_version": VERSION, "config_id": identity, "replay_file": context.replay_file,
                              "beatmap_hash": context.beatmap_hash, "performance_category": source.performance_category,
                              "pn_key_index": pn_index - 1, "pn_key_id": pn_id, "historical_pn": int(pn_index == 0),
                              "layout_seed": seed, "layout_key_id": key_id(layout_key), "alpha": alpha, "N": N_VALUE,
                              "payload_fraction": PAYLOAD_FRACTION, "coded_bits": len(encoded), "useful_bits": len(useful),
                              "message_sha256": array_hash(useful), "encoded_sha256": array_hash(encoded), "replay_sha256": replay_sha}
                    for word in range(len(encoded) // 8):
                        cs, us = slice(word*8, word*8+8), slice(word*4, word*4+4)
                        syndrome, parity = secded_syndrome(received[cs])
                        decision = corrected_bit_is_not_minimum_decision(str(statuses[word]), syndrome, parity, np.abs(corr[cs]))
                        correct = bool(np.array_equal(decoded[us], useful[us])); decisions.append(decision); words_correct.append(correct)
                        rank = np.nan
                        if str(statuses[word]) == "corrected_single":
                            pos = syndrome - 1 if syndrome else 7
                            rank = int(np.argsort(np.argsort(np.abs(corr[cs]), kind="stable"), kind="stable")[pos])
                        buffers["codeword"].append({**common, "codeword_index": word, "hard_status": str(statuses[word]),
                            "raw_error_count": int(np.sum(received[cs] != encoded[cs])), "word_correct": int(correct),
                            "integrity_accepted": int(decision.accepted), "integrity_reason": decision.reason,
                            "corrected_bit_confidence_rank": rank})
                    accepted, correct = all(d.accepted for d in decisions), all(words_correct)
                    outcome = "REJECT" if not accepted else ("CORRECT_ACCEPT" if correct else "WRONG_ACCEPT")
                    buffers["message"].append({**common, "outcome": outcome,
                        "correct_accept": int(outcome == "CORRECT_ACCEPT"), "wrong_accept": int(outcome == "WRONG_ACCEPT"),
                        "reject": int(outcome == "REJECT"), "post_ecc_bit_errors": int(np.sum(decoded != useful)),
                        "post_ecc_ber": float(np.mean(decoded != useful))})
                    requested, active = int(diagnostics["requested_carriers"]), int(diagnostics["active_carriers"])
                    buffers["physical"].append({**common, "raw_bit_errors": int(np.sum(received != encoded)),
                        "raw_ber": float(np.mean(received != encoded)), "active_carrier_fraction": active/requested,
                        **diagnostics})
                    bit_rows, _ = bit_level_diagnostics(context=context, roundtrip_residuals=roundtrip, message=encoded,
                        alpha=alpha, n_frames_per_bit=N_VALUE, pn_key=pn_key, layout_key=layout_key,
                        hit_margin_ms=float(adaptive["hit_margin_ms"]))
                    buffers["bit"].extend({**common, **row} for row in bit_rows)
                    for label, values in ((0, clean_features), (1, residual_timing_features(roundtrip))):
                        buffers["features"].append({**common, "label": label, "feature_version": FEATURE_VERSION, **values})
                    wrong_received = extract_bipolar_message_with_layout(roundtrip, wrong_key["key"], layout_key, N_VALUE, len(encoded), "distributed")
                    wrong_corr = message_correlations_with_layout(roundtrip, wrong_key["key"], layout_key, N_VALUE, len(encoded), "distributed")
                    buffers["wrong"].append({**common, "wrong_pn_key_id": wrong_key["key_id"],
                        "wrong_key_bit_errors": int(np.sum(wrong_received != encoded)), "wrong_key_ber": float(np.mean(wrong_received != encoded)),
                        "wrong_key_mean_abs_correlation": float(np.mean(np.abs(wrong_corr)))})
                    new += 1
            if replay_number % 5 == 0:
                flush(args.output_dir, buffers); print(f"physical {replay_number}/{len(selected)} new={new}", flush=True)
    flush(args.output_dir, buffers)
    return new


def flush(output: Path, buffers: dict[str, list[dict]]) -> None:
    specs = {"physical": ("physical_results.csv", ("config_id",)), "message": ("message_results.csv", ("config_id",)),
             "codeword": ("codeword_results.csv", ("config_id", "codeword_index")), "bit": ("bit_results.csv", ("config_id", "bit_index")),
             "features": ("timing_features.csv", ("config_id", "label")), "wrong": ("wrong_key_sanity.csv", ("config_id",))}
    for name, (filename, keys) in specs.items():
        append_unique(output / filename, buffers[name], keys); buffers[name].clear()


def diagnostics(args: argparse.Namespace, selection: pd.DataFrame, pn_config: dict, new: int) -> None:
    physical = pd.read_csv(args.output_dir / "physical_results.csv")
    eligible = int(selection.eligible_7_5.sum()); expected = eligible * len(LAYOUT_KEYS) * (1 + len(pn_config["new_keys"]))
    rows = [{"check": "new_configs_this_run", "value": new}, {"check": "incoming_replays", "value": len(selection)},
            {"check": "eligible_replays", "value": eligible}, {"check": "eligible_maps", "value": selection.loc[selection.eligible_7_5 == 1, "beatmap_hash"].nunique()},
            {"check": "expected_configs", "value": expected}, {"check": "physical_configs", "value": len(physical)},
            {"check": "duplicate_configs", "value": int(physical.config_id.duplicated().sum())},
            {"check": "pn_keys_per_replay_layout", "value": int(physical.groupby(["replay_file", "layout_seed"]).pn_key_id.nunique().min())}]
    pd.DataFrame(rows).to_csv(args.output_dir / "diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT); parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--selection", type=Path, default=OUTPUT / "pilot_selection.csv")
    parser.add_argument("--pn-config", type=Path, default=CONFIG_DIR / "pn_key_robustness_v1.json")
    parser.add_argument("--lock", type=Path, default=OUTPUT / "experiment_lock.json")
    parser.add_argument("--adaptive-policy", type=Path, default=CONFIG_DIR / "adaptive_alpha_sender_local_v2.json")
    parser.add_argument("--integrity-rule", type=Path, default=CONFIG_DIR / "ecc_integrity_rejection_v1.json")
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--limit-replays", type=int); args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    pn_config, adaptive, selection = load_locked(args)
    new = generate(args, pn_config, adaptive, selection); diagnostics(args, selection, pn_config, new)
    print(f"PN robustness physical complete new_configs={new}")


if __name__ == "__main__":
    main()
