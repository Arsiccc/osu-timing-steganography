"""Run the locked host/message-aware PN-selection physical holdout."""

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
from osu_stego.stego.ecc import encode_hamming_8_4_secded
from osu_stego.stego.host_aware_pn import (
    MESSAGE_FAMILIES,
    decode_with_frozen_integrity,
    deterministic_message_family,
    predicted_margins,
    select_candidate,
    summarize_margins,
)
from osu_stego.stego.payload_layout import (
    extract_bipolar_message_with_layout,
    message_correlations_with_layout,
)
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import array_hash
from scripts.experiments.run_layout_comparison import build_physical_stego, key_id
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    prepare_replay,
)
from scripts.experiments.run_pn_key_robustness import append_unique


VERSION = "host-aware-pn-selection-v1"
OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"
N_VALUE = 8
PAYLOAD_FRACTION = 0.075
K_VALUES = (1, 2, 4, 8)
LAYOUT_INDICES = (0, 1, 2)


def stable_id(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def load_locked(args: argparse.Namespace) -> tuple[dict, dict, pd.DataFrame]:
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    checks = (
        (args.pn_config, "pn_candidates_sha256"),
        (args.selector, "selector_sha256"),
        (args.messages, "message_families_sha256"),
        (args.holdout, "holdout_selection_sha256"),
        (args.adaptive_policy, "adaptive_policy_sha256"),
        (args.integrity_rule, "integrity_rule_sha256"),
        (args.map_offsets, "map_offsets_sha256"),
    )
    for path, field in checks:
        if file_sha256(path) != lock[field]:
            raise RuntimeError(f"Locked hash mismatch: {path}")
    for path, field in (
        (Path(__file__), "runner"),
        (Path("osu_stego/stego/host_aware_pn.py"), "host_aware_module"),
    ):
        if file_sha256(path) != lock["source_sha256"][field]:
            raise RuntimeError(f"Locked source mismatch: {path}")
    pn_config = json.loads(args.pn_config.read_text(encoding="utf-8"))
    return pn_config, load_sender_local_policy(args.adaptive_policy), pd.read_csv(args.holdout)


def physical_decode(
    roundtrip: np.ndarray,
    pn_key: str,
    layout_key: str,
    coded_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    received = extract_bipolar_message_with_layout(
        roundtrip, pn_key, layout_key, N_VALUE, coded_length, "distributed"
    )
    correlations = message_correlations_with_layout(
        roundtrip, pn_key, layout_key, N_VALUE, coded_length, "distributed"
    )
    return received, correlations


def selection_scores(
    residuals: np.ndarray,
    encoded: np.ndarray,
    candidates: list[dict],
    layout_key: str,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    margins: dict[str, np.ndarray] = {}
    scores = {}
    for candidate in candidates:
        candidate_id = str(candidate["key_id"])
        values = predicted_margins(
            residuals, encoded, str(candidate["key"]), layout_key, alpha, N_VALUE
        )
        margins[candidate_id] = values
        scores[candidate_id] = summarize_margins(values)
    return margins, scores


def blind_decision(
    roundtrip: np.ndarray,
    candidates: list[dict],
    layout_key: str,
    encoded_length: int,
) -> tuple[str, np.ndarray | None, str, list[dict]]:
    """Operational rule uses only received replay and candidate hypotheses."""
    diagnostics = []
    accepted_messages: list[np.ndarray] = []
    for candidate in candidates:
        received, correlations = physical_decode(
            roundtrip, str(candidate["key"]), layout_key, encoded_length
        )
        decoded = decode_with_frozen_integrity(received, correlations)
        diagnostics.append({
            "hypothesis_pn_id": candidate["key_id"],
            "integrity_accepted": int(decoded.accepted),
            "integrity_reason": decoded.rejection_reason,
            "mean_abs_correlation": float(np.mean(np.abs(correlations))),
            "decoded_message_sha256": array_hash(decoded.useful_bits),
            "decoded_useful_bits": "".join("1" if bit > 0 else "0" for bit in decoded.useful_bits),
        })
        if decoded.accepted:
            accepted_messages.append(decoded.useful_bits)
    if not accepted_messages:
        return "REJECT", None, "zero_candidate", diagnostics
    unique = {array_hash(message): message for message in accepted_messages}
    if len(unique) > 1:
        return "REJECT", None, "ambiguous_candidate", diagnostics
    return "ACCEPT", next(iter(unique.values())), "accepted_common_message", diagnostics


def completed_units(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["unit_id"].astype(str))


def flush(output: Path, buffers: dict[str, list[dict]]) -> None:
    specs = {
        "scores": ("candidate_scores.csv", ("unit_id", "pn_key_id")),
        "physical": ("physical_results.csv", ("physical_id",)),
        "bits": ("bit_results.csv", ("physical_id", "bit_index")),
        "features": ("timing_features.csv", ("physical_id", "label")),
        "known": ("index_known_results.csv", ("condition_id",)),
        "blind": ("blind_decoder_results.csv", ("condition_id",)),
        "blind_diag": ("blind_candidate_diagnostics.csv", ("condition_id", "hypothesis_pn_id")),
        "units": ("completed_units.csv", ("unit_id",)),
    }
    for name, (filename, keys) in specs.items():
        append_unique(output / filename, buffers[name], keys)
        buffers[name].clear()


def run(args: argparse.Namespace, pn_config: dict, policy: dict, holdout: pd.DataFrame) -> int:
    if args.limit_replays is not None:
        holdout = holdout.sort_values("selection_order").head(args.limit_replays)
    candidates = list(pn_config["candidates"])
    candidate_by_id = {str(value["key_id"]): value for value in candidates}
    done = completed_units(args.output_dir / "completed_units.csv")
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    buffers = {name: [] for name in (
        "scores", "physical", "bits", "features", "known", "blind",
        "blind_diag", "units",
    )}
    new_units = 0
    with tempfile.TemporaryDirectory(prefix="host_aware_pn_") as temp_name:
        temp_dir = Path(temp_name)
        for replay_number, source in enumerate(holdout.itertuples(index=False), 1):
            context = prepare_replay(source, args.dataset_dir, beatmaps, offsets)
            alpha = float(alpha_from_replay(policy, context.replay))
            replay_sha = file_sha256(context.osr_path)
            clean_features = residual_timing_features(context.original_residuals)
            for family in MESSAGE_FAMILIES:
                useful = deterministic_message_family(
                    family, int(source.useful_bits), context.replay_file, context.beatmap_hash
                )
                encoded = encode_hamming_8_4_secded(useful)
                for layout_seed in LAYOUT_INDICES:
                    layout_key = LAYOUT_KEYS[layout_seed]
                    unit_id = stable_id(VERSION, replay_sha, family, layout_seed, array_hash(encoded))
                    if unit_id in done:
                        continue
                    common = {
                        "experiment_version": VERSION,
                        "unit_id": unit_id,
                        "replay_file": context.replay_file,
                        "beatmap_hash": context.beatmap_hash,
                        "performance_category": source.performance_category,
                        "message_family": family,
                        "layout_seed": layout_seed,
                        "layout_key_id": key_id(layout_key),
                        "alpha": alpha,
                        "N": N_VALUE,
                        "payload_fraction": PAYLOAD_FRACTION,
                        "coded_bits": len(encoded),
                        "useful_bits": len(useful),
                        "message_sha256": array_hash(useful),
                        "encoded_sha256": array_hash(encoded),
                        "replay_sha256": replay_sha,
                    }
                    margins, scores = selection_scores(
                        context.original_residuals, encoded, candidates, layout_key, alpha
                    )
                    for candidate in candidates:
                        candidate_id = str(candidate["key_id"])
                        score = scores[candidate_id]
                        buffers["scores"].append({
                            **common,
                            "pn_index": candidate["index"],
                            "pn_key_id": candidate_id,
                            "effective_seed_32": candidate["effective_seed_32"],
                            "nonpositive_count": score.nonpositive_count,
                            "minimum_margin": score.minimum_margin,
                            "mean_margin": score.mean_margin,
                            "margin_sha256": array_hash(margins[candidate_id]),
                        })
                    selected_by_k = {
                        k_value: select_candidate([
                            (str(candidate["key_id"]), scores[str(candidate["key_id"])])
                            for candidate in candidates[:k_value]
                        ])
                        for k_value in K_VALUES
                    }
                    if selected_by_k[1] != str(candidates[0]["key_id"]):
                        raise RuntimeError("K=1 failed to select candidate zero.")

                    physical_cache: dict[str, dict] = {}
                    for selected_id in sorted(set(selected_by_k.values())):
                        candidate = candidate_by_id[selected_id]
                        physical_id = stable_id(unit_id, selected_id)
                        _, roundtrip, physical_diag = build_physical_stego(
                            context, encoded, alpha, N_VALUE, "distributed",
                            str(candidate["key"]), layout_key, float(policy["hit_margin_ms"]),
                            temp_dir / f"{physical_id}.osr",
                        )
                        received, correlations = physical_decode(
                            roundtrip, str(candidate["key"]), layout_key, len(encoded)
                        )
                        decoded = decode_with_frozen_integrity(received, correlations)
                        correct = bool(np.array_equal(decoded.useful_bits, useful))
                        outcome = "REJECT" if not decoded.accepted else (
                            "CORRECT_ACCEPT" if correct else "WRONG_ACCEPT"
                        )
                        physical_cache[selected_id] = {
                            "physical_id": physical_id,
                            "roundtrip": roundtrip,
                            "received": received,
                            "correlations": correlations,
                            "decoded": decoded,
                            "outcome": outcome,
                            "physical_diag": physical_diag,
                        }
                        requested = int(physical_diag["requested_carriers"])
                        active = int(physical_diag["active_carriers"])
                        buffers["physical"].append({
                            **common, "physical_id": physical_id,
                            "selected_pn_id": selected_id,
                            "selected_pn_index": candidate["index"],
                            "raw_bit_errors": int(np.sum(received != encoded)),
                            "raw_ber": float(np.mean(received != encoded)),
                            "post_ecc_bit_errors": int(np.sum(decoded.useful_bits != useful)),
                            "post_ecc_ber": float(np.mean(decoded.useful_bits != useful)),
                            "index_known_outcome": outcome,
                            "active_carrier_fraction": active / requested if requested else 0.0,
                            **physical_diag,
                        })
                        bit_rows, _ = bit_level_diagnostics(
                            context=context, roundtrip_residuals=roundtrip, message=encoded,
                            alpha=alpha, n_frames_per_bit=N_VALUE,
                            pn_key=str(candidate["key"]), layout_key=layout_key,
                            hit_margin_ms=float(policy["hit_margin_ms"]),
                        )
                        for row, predicted in zip(bit_rows, margins[selected_id].tolist()):
                            buffers["bits"].append({
                                **common, "physical_id": physical_id,
                                "selected_pn_id": selected_id,
                                "predicted_margin": predicted, **row,
                            })
                        for label, values in (
                            (0, clean_features), (1, residual_timing_features(roundtrip))
                        ):
                            buffers["features"].append({
                                **common, "physical_id": physical_id,
                                "selected_pn_id": selected_id, "label": label,
                                "feature_version": FEATURE_VERSION, **values,
                            })

                    for k_value, selected_id in selected_by_k.items():
                        cached = physical_cache[selected_id]
                        condition_id = stable_id(unit_id, k_value)
                        received = cached["received"]
                        decoded = cached["decoded"]
                        buffers["known"].append({
                            **common, "condition_id": condition_id, "K": k_value,
                            "selected_pn_id": selected_id,
                            "selected_pn_index": candidate_by_id[selected_id]["index"],
                            "physical_id": cached["physical_id"],
                            "raw_bit_errors": int(np.sum(received != encoded)),
                            "raw_ber": float(np.mean(received != encoded)),
                            "post_ecc_bit_errors": int(np.sum(decoded.useful_bits != useful)),
                            "post_ecc_ber": float(np.mean(decoded.useful_bits != useful)),
                            "outcome": cached["outcome"],
                            "correct_accept": int(cached["outcome"] == "CORRECT_ACCEPT"),
                            "wrong_accept": int(cached["outcome"] == "WRONG_ACCEPT"),
                            "reject": int(cached["outcome"] == "REJECT"),
                            "pn_index_side_information_bits": int(np.ceil(np.log2(k_value))),
                        })
                        blind_status, blind_message, blind_reason, hypothesis_rows = blind_decision(
                            cached["roundtrip"], candidates[:k_value], layout_key, len(encoded)
                        )
                        blind_outcome = "REJECT"
                        if blind_status == "ACCEPT":
                            blind_outcome = (
                                "CORRECT_ACCEPT" if np.array_equal(blind_message, useful)
                                else "WRONG_ACCEPT"
                            )
                        accepted_count = sum(row["integrity_accepted"] for row in hypothesis_rows)
                        buffers["blind"].append({
                            **common, "condition_id": condition_id, "K": k_value,
                            "selected_pn_id": selected_id,
                            "physical_id": cached["physical_id"],
                            "outcome": blind_outcome,
                            "decision_reason": blind_reason,
                            "correct_accept": int(blind_outcome == "CORRECT_ACCEPT"),
                            "wrong_accept": int(blind_outcome == "WRONG_ACCEPT"),
                            "reject": int(blind_outcome == "REJECT"),
                            "accepted_hypotheses": accepted_count,
                            "ambiguous_reject": int(blind_reason == "ambiguous_candidate"),
                            "zero_candidate_reject": int(blind_reason == "zero_candidate"),
                        })
                        for hypothesis in hypothesis_rows:
                            hypothesis_message = hypothesis.pop("decoded_useful_bits")
                            buffers["blind_diag"].append({
                                **common, "condition_id": condition_id, "K": k_value,
                                "selected_pn_id": selected_id,
                                "physical_id": cached["physical_id"],
                                **hypothesis,
                                "is_selected_hypothesis": int(
                                    hypothesis["hypothesis_pn_id"] == selected_id
                                ),
                                "message_matches_truth": int(
                                    hypothesis_message == "".join(
                                        "1" if bit > 0 else "0" for bit in useful
                                    )
                                ),
                            })
                    buffers["units"].append({**common})
                    new_units += 1
            flush(args.output_dir, buffers)
            print(
                f"physical {replay_number}/{len(holdout)} replays | new_units={new_units}",
                flush=True,
            )
    return new_units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--pn-config", type=Path, default=CONFIG_DIR / "host_aware_pn_candidates_v1.json")
    parser.add_argument("--selector", type=Path, default=OUTPUT / "frozen_selector.json")
    parser.add_argument("--messages", type=Path, default=OUTPUT / "message_families.json")
    parser.add_argument("--holdout", type=Path, default=OUTPUT / "holdout_selection.csv")
    parser.add_argument("--lock", type=Path, default=OUTPUT / "experiment_lock.json")
    parser.add_argument("--adaptive-policy", type=Path, default=CONFIG_DIR / "adaptive_alpha_sender_local_v2.json")
    parser.add_argument("--integrity-rule", type=Path, default=CONFIG_DIR / "ecc_integrity_rejection_v1.json")
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--limit-replays", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, policy, holdout = load_locked(args)
    new_units = run(args, config, policy, holdout)
    print(f"Host-aware physical run complete; new_units={new_units}")


if __name__ == "__main__":
    main()
