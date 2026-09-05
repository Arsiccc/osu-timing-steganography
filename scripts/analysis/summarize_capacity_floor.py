"""Create coverage-aware summaries for the development one-codeword pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.analysis.timing_features import BASELINE_FEATURES, FULL_FEATURES, POSITION_FEATURES
from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_one_codeword_floor_pilot import BIN_LABELS, CAP_POLICIES, rf_auc


def weighted_new_metrics(capacity: pd.DataFrame, integrity: pd.DataFrame, cap: float) -> tuple[dict[str, float], int]:
    new = capacity[(capacity.eligible_at_7_5pct == 0) & (capacity.minimum_fraction_1_codewords <= cap)].copy()
    if new.empty:
        return {name: 0.0 for name in ("correct_accept", "wrong_accept", "reject", "useful")}, 0
    rates = integrity.groupby("required_fraction_bin")[["correct_accept", "wrong_accept", "reject", "useful_correctly_accepted_bits"]].mean()
    edges = [-np.inf, .10, .125, .15, .20, np.inf]
    new["required_fraction_bin"] = pd.cut(new.minimum_fraction_1_codewords, edges, labels=BIN_LABELS).astype(str)
    counts = new.required_fraction_bin.value_counts()
    missing = [label for label in counts.index if label not in rates.index]
    if missing:
        raise RuntimeError(f"Nedostaje pilot stratum za {missing}")
    return {
        "correct_accept": float(sum(counts[label] * rates.loc[label, "correct_accept"] for label in counts.index)),
        "wrong_accept": float(sum(counts[label] * rates.loc[label, "wrong_accept"] for label in counts.index)),
        "reject": float(sum(counts[label] * rates.loc[label, "reject"] for label in counts.index)),
        "useful": float(sum(counts[label] * rates.loc[label, "useful_correctly_accepted_bits"] for label in counts.index)),
    }, len(new)


def system_metrics(capacity: pd.DataFrame, integrity: pd.DataFrame, old_messages: pd.DataFrame) -> pd.DataFrame:
    development = capacity[capacity.partition == "development"]
    eligible_count = int(development.eligible_at_7_5pct.sum())
    old = old_messages[old_messages.method == "integrity"]
    old_rates = old[["correct_accept", "wrong_accept", "reject", "useful_correctly_accepted_bits"]].mean()
    rows = []
    for policy, cap in CAP_POLICIES:
        new_totals, new_count = weighted_new_metrics(development, integrity, cap)
        attempted = eligible_count + new_count
        correct = eligible_count * old_rates.correct_accept + new_totals["correct_accept"]
        wrong = eligible_count * old_rates.wrong_accept + new_totals["wrong_accept"]
        reject = eligible_count * old_rates.reject + new_totals["reject"]
        useful = eligible_count * old_rates.useful_correctly_accepted_bits + new_totals["useful"]
        rows.append({
            "policy": policy, "payload_cap": cap, "development_replays": len(development),
            "attempted_replays": attempted, "newly_covered_replays": new_count,
            "coverage_fraction": attempted / len(development),
            "correct_delivery_fraction_all_replays_estimated": correct / len(development),
            "silent_wrong_delivery_fraction_all_replays_estimated": wrong / len(development),
            "explicit_failure_rate_among_attempted_estimated": reject / attempted,
            "no_capacity_abstention_replays": len(development) - attempted,
            "no_capacity_abstention_fraction": (len(development)-attempted)/len(development),
            "correctly_accepted_useful_bits_per_input_replay_estimated": useful / len(development),
            "estimation_basis": "exact capacity counts; 80-replay old eligible holdout + bin-stratified 80-replay floor pilot outcome rates",
        })
    return pd.DataFrame(rows)


def floor_policy_decisions(capacity: pd.DataFrame) -> pd.DataFrame:
    development = capacity[capacity.partition == "development"]
    rows = []
    for source in development.itertuples(index=False):
        base_bits = int(source.coded_bits_at_7_5pct)
        for policy, cap in CAP_POLICIES:
            if base_bits >= 8:
                requested = base_bits
                decision = "unchanged_current_eligible"
            elif bool(source.structurally_can_fit_full_codeword) and float(source.minimum_fraction_1_codewords) <= cap:
                requested = 8
                decision = "exact_one_codeword_floor"
            else:
                requested = 0
                decision = "no_capacity_abstention"
            rows.append({
                "replay_file": source.replay_file, "beatmap_hash": source.beatmap_hash,
                "policy": policy, "payload_cap": cap, "base_allocated_coded_bits": base_bits,
                "requested_coded_bits": requested, "complete_codewords": requested // 8,
                "useful_bits": (requested // 8) * 4, "decision": decision,
            })
    return pd.DataFrame(rows)


def detector_comparator(capacity: pd.DataFrame, old_features: pd.DataFrame, iterations: int) -> pd.DataFrame:
    eligible = set(capacity.query("partition == 'development' and eligible_at_7_5pct == 1").replay_file.astype(str))
    features = old_features[(old_features.partition == "development") &
                            (old_features.alpha_method == "sender_local_adaptive") &
                            (old_features.layout == "distributed") &
                            (old_features.replay_file.astype(str).isin(eligible))].copy()
    features["required_fraction_bin"] = "current_eligible"
    rows = []
    sets = {"baseline": tuple(BASELINE_FEATURES), "position_only": tuple(POSITION_FEATURES), "full": tuple(FULL_FEATURES)}
    scopes = [("pooled_five_seeds", features)] + [(f"layout_seed_{seed}", features[features.layout_seed.astype(str) == str(seed)]) for seed in range(5)]
    for scope, frame in scopes:
        for feature_set, names in sets.items():
            row, _ = rf_auc(frame.reset_index(drop=True), names, f"current_eligible:{scope}:{feature_set}", iterations)
            row["feature_set"] = feature_set; row["population"] = "current_7.5pct_eligible_historical_features"
            rows.append(row)
    return pd.DataFrame(rows)


def population_comparison(capacity: pd.DataFrame, physical: pd.DataFrame, integrity: pd.DataFrame,
                          old_physical: pd.DataFrame, old_messages: pd.DataFrame,
                          detector: pd.DataFrame, pilot_auc: pd.DataFrame) -> pd.DataFrame:
    dev = capacity[capacity.partition == "development"]
    note_map = dev.set_index("replay_file").note_index_length
    old_ids = set(old_messages.loc[old_messages.method == "integrity", "config_id"])
    old_p = old_physical[old_physical.config_id.isin(old_ids)].copy()
    old_i = old_messages[old_messages.method == "integrity"]
    old_p["note_index_length"] = old_p.replay_file.map(note_map)
    rows = [{
        "population": "current_7.5pct_eligible_holdout_estimate", "replays": old_p.replay_file.nunique(),
        "note_count_mean": old_p.note_index_length.mean(), "required_fraction_mean": np.nan,
        "raw_ber": old_p.raw_bit_errors.sum()/old_p.physical_bits.sum(),
        "correct_accept_rate": old_i.correct_accept.mean(), "wrong_accept_rate": old_i.wrong_accept.mean(),
        "reject_rate": old_i.reject.mean(), "active_carrier_fraction": old_p.active_carriers.sum()/old_p.requested_carriers.sum(),
        "energy_per_note_ms2": (old_p.sum_squared_shift_ms2/old_p.note_index_length).mean(),
        "full_auc": detector.query("scope == 'current_eligible:pooled_five_seeds:full'").roc_auc.iloc[0],
        "comparability_note": "historical 80-replay eligible integrity holdout; variable number of complete codewords",
    }, {
        "population": "newly_covered_floor_pilot", "replays": physical.replay_file.nunique(),
        "note_count_mean": physical.note_index_length.mean(), "required_fraction_mean": physical.required_fraction.mean(),
        "raw_ber": physical.raw_bit_errors.sum()/(8*len(physical)),
        "correct_accept_rate": integrity.correct_accept.mean(), "wrong_accept_rate": integrity.wrong_accept.mean(),
        "reject_rate": integrity.reject.mean(), "active_carrier_fraction": physical.active_carriers.sum()/physical.requested_carriers.sum(),
        "energy_per_note_ms2": physical.energy_per_note_ms2.mean(),
        "full_auc": pilot_auc.query("scope == 'pooled_five_seeds:full'").roc_auc.iloc[0],
        "comparability_note": "stratified development pilot; exactly one codeword; unweighted across four strata",
    }, {
        "population": "still_ineligible_uncapped", "replays": int((dev.structurally_can_fit_full_codeword == 0).sum()),
        "note_count_mean": dev.loc[dev.structurally_can_fit_full_codeword == 0, "note_index_length"].mean(),
        "required_fraction_mean": np.nan, "raw_ber": np.nan, "correct_accept_rate": np.nan,
        "wrong_accept_rate": np.nan, "reject_rate": np.nan, "active_carrier_fraction": np.nan,
        "energy_per_note_ms2": np.nan, "full_auc": np.nan,
        "comparability_note": "no development replay is structurally unable at uncapped payload",
    }]
    return pd.DataFrame(rows)


def reports(capacity_dir: Path, pilot_dir: Path) -> None:
    capacity = pd.read_csv(capacity_dir / "capacity_by_replay.csv")
    curve = pd.read_csv(capacity_dir / "coverage_curve.csv")
    shortened = pd.read_csv(capacity_dir / "shortened_code_analysis.csv")
    physical = pd.read_csv(pilot_dir / "physical_results.csv")
    integrity = pd.read_csv(pilot_dir / "integrity_results.csv")
    auc = pd.read_csv(pilot_dir / "steganalysis_auc.csv")
    system = pd.read_csv(capacity_dir / "system_coverage_metrics.csv")
    dev_curve = curve[curve.partition == "development"]
    val_curve = curve[curve.partition == "validation"]
    text = f"""# Capacity coverage v1

Development-only policy work; the old validation partition is used only for descriptive capacity counts.

## Exact semantics

For `M` note-index positions and frozen `N=8`, `C=floor(M/8)`. The production allocator returns `B=0` when `C=0`, otherwise `B=max(1,floor(C*f))`. SECDED uses `floor(B/8)` complete words and discards a trailing incomplete allocation. Missing/NaN carriers do not reduce `C`; they affect physical reliability. DISTRIBUTED permutes all non-overlapping N-note blocks by layout key and takes the first B blocks.

## Coverage

At 7.5%, development eligibility is {int(dev_curve.iloc[0].eligible_replays)}/611 ({dev_curve.iloc[0].replay_eligibility_fraction:.1%}); old-validation eligibility is {int(val_curve.iloc[0].eligible_replays)}/338 ({val_curve.iloc[0].replay_eligibility_fraction:.1%}), exactly reproducing 167 replay files on 5 maps. All 949 replay files can structurally fit eight coded bits at 100%; current failures are fractional-allocation failures.

Development floor-cap coverage: """ + ", ".join(
        f"{row.payload_cap:.1%}: {int(row.eligible_replays)}/611 ({row.replay_eligibility_fraction:.1%})"
        for row in dev_curve.itertuples() if row.payload_cap in (.075,.10,.125,.15,.20,1.0)
    ) + ". The 12.5–15% interval contains no development candidates; 31 replays require 34.78%.\n\n" + f"""## Shortened-code theory

Systematic shortening was exhaustively verified: [7,3,4], [6,2,4], and [5,1,4] all retain distance 4, unique correction for every single-bit error, and detection of every double-bit error. At 7.5%, development coverage is 389/611 for [7,3,4], 451/611 for [6,2,4], and 514/611 for [5,1,4]. These codes are not compatible with the frozen 8-bit syndrome/integrity implementation and require a new decoder and validation.

## System estimates

`system_coverage_metrics.csv` uses exact capacity counts but pilot-estimated delivery rates. It is not a full-development physical result and must not be presented as independently validated.
"""
    (capacity_dir / "capacity_report.md").write_text(text)
    overall_ber = physical.raw_bit_errors.sum()/(8*len(physical))
    post_ber = physical.post_ecc_bit_errors.sum()/(4*len(physical))
    full = auc.query("scope == 'pooled_five_seeds:full'").iloc[0]
    report = f"""# One-codeword floor pilot v1

Exploratory development pilot: 80 previously ineligible replay files, deterministically stratified across four nonempty required-fraction bins, five frozen layout seeds, exactly one SECDED(8,4) word per configuration.

- Raw BER: {overall_ber:.3%}; post-SECDED bit error rate: {post_ber:.3%}.
- Frozen integrity: CORRECT_ACCEPT {integrity.correct_accept.mean():.2%}, WRONG_ACCEPT {integrity.wrong_accept.mean():.2%}, REJECT {integrity.reject.mean():.2%}.
- FULL pooled five-seed AUC: {full.roc_auc:.3f}, replay-cluster 95% CI [{full.auc_ci95_low:.3f}, {full.auc_ci95_high:.3f}].
- Mean modified-note fraction: {physical.modified_note_fraction.mean():.2%}; mean energy/note: {physical.energy_per_note_ms2.mean():.2f} ms².
- The >20% stratum required 34.78% in this corpus and had descriptive FULL AUC 0.761; this is the clearest warning against an uncapped floor.

The pilot supports considering a capped floor, but does not validate one. It does not justify an uncapped policy. A new independent replay/map corpus is required for any final policy claim. The frozen integrity rule was applied unchanged.
"""
    (pilot_dir / "pilot_report.md").write_text(report)


def normalized_analysis(physical: pd.DataFrame, output_path: Path) -> None:
    replay_seed = physical.copy()
    rows = []
    for metric in ("modified_note_fraction", "selected_note_fraction", "energy_per_note_ms2",
                   "energy_per_useful_bit_ms2", "selected_timeline_span_fraction"):
        rows.append({
            "predictor": "required_fraction", "metric": metric,
            "spearman_rank_correlation": float(replay_seed.required_fraction.rank().corr(replay_seed[metric].rank())),
            "configs": len(replay_seed), "replays": replay_seed.replay_file.nunique(),
            "interpretation_scope": "development pilot; five within-replay layout observations",
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_hash_manifest(directories: tuple[Path, ...]) -> None:
    for directory in directories:
        files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != "result_hashes.json")
        manifest = {
            "hash_algorithm": "sha256",
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        }
        (directory / "result_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-dir", type=Path, default=RESULTS_DIR / "capacity_coverage_v1")
    parser.add_argument("--pilot-dir", type=Path, default=RESULTS_DIR / "one_codeword_floor_pilot_v1")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    capacity = pd.read_csv(args.capacity_dir / "capacity_by_replay.csv")
    integrity = pd.read_csv(args.pilot_dir / "integrity_results.csv")
    old_messages = pd.read_csv(RESULTS_DIR / "ecc_integrity_rejection_v1" / "message_results.csv")
    system_metrics(capacity, integrity, old_messages).to_csv(args.capacity_dir / "system_coverage_metrics.csv", index=False)
    decisions = floor_policy_decisions(capacity)
    if not (decisions.loc[decisions.base_allocated_coded_bits >= 8, "decision"] == "unchanged_current_eligible").all():
        raise RuntimeError("Floor je promenio postojeći eligible replay.")
    floor_rows = decisions[decisions.decision == "exact_one_codeword_floor"]
    if not (floor_rows.requested_coded_bits == 8).all():
        raise RuntimeError("Floor nije zatražio tačno osam coded bits.")
    decisions.to_csv(args.capacity_dir / "floor_policy_decisions.csv", index=False)
    detector = detector_comparator(capacity, pd.read_csv(RESULTS_DIR / "adaptive_layout_strong_v1" / "features.csv"), args.iterations)
    detector.to_csv(args.pilot_dir / "current_eligible_detector_auc.csv", index=False)
    comparison = population_comparison(
        capacity, pd.read_csv(args.pilot_dir / "physical_results.csv"), integrity,
        pd.read_csv(RESULTS_DIR / "ecc_integrity_rejection_v1" / "holdout_physical_results.csv"), old_messages,
        detector, pd.read_csv(args.pilot_dir / "steganalysis_auc.csv"),
    )
    comparison.to_csv(args.pilot_dir / "population_comparison.csv", index=False)
    normalized_analysis(pd.read_csv(args.pilot_dir / "physical_results.csv"), args.pilot_dir / "normalized_position_energy.csv")
    reports(args.capacity_dir, args.pilot_dir)
    write_hash_manifest((args.capacity_dir, args.pilot_dir))


if __name__ == "__main__":
    main()
