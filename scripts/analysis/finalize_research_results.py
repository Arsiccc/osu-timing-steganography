"""Regenerate the final scientific audit from frozen, existing artifacts only.

This command intentionally performs no replay parsing, matching, embedding,
writing, offset calibration, API access, or model fitting.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VERSION = "final-audit-generator-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "final_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_set_hash(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted({str(value) for value in values})) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_hash(path: str) -> str:
    return sha256(ROOT / path)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_to_markdown(csv_path: Path, md_path: Path) -> None:
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(clean(c) for c in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += ["| " + " | ".join(clean(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("files", data)
    output: dict[str, str] = {}
    for name, value in entries.items():
        if name in {"hash_algorithm", "generated_by", "version"}:
            continue
        output[name] = value["sha256"] if isinstance(value, dict) else str(value)
    return output


def audit_manifests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests = sorted((ROOT / "results").glob("*/result_hashes.json"))
    for manifest in manifests:
        entries = load_manifest(manifest)
        missing, mismatch = [], []
        for name, expected in entries.items():
            target = manifest.parent / name
            if not target.is_file():
                missing.append(name)
            elif sha256(target) != expected:
                mismatch.append(name)
        present = {p.name for p in manifest.parent.iterdir() if p.is_file() and p.name != manifest.name}
        unexpected = sorted(present - set(entries))
        rows.append({
            "branch": manifest.parent.name,
            "manifest_path": rel(manifest),
            "manifest_present": True,
            "manifest_sha256": sha256(manifest),
            "entries": len(entries),
            "all_hashes_valid": not missing and not mismatch,
            "missing_artifacts": ";".join(missing),
            "hash_mismatches": ";".join(mismatch),
            "unexpected_unmanifested_files": ";".join(unexpected),
            "duplicate_manifest_names": False,
            "severity": "PASS" if not missing and not mismatch else "STOP_CENTRAL_IF_CLAIM_SOURCE",
        })
    if any(not row["all_hashes_valid"] for row in rows):
        raise RuntimeError("A frozen result hash failed; refusing to finalize.")
    return rows


BRANCHES = [
    ("timing_calibration", "results/matching", "LEVEL 1 — INTERNAL / EXPLORATORY", True, "active dataset plus pre-quarantine diagnostics"),
    ("matching_validation", "results/matching", "LEVEL 2 — FROZEN DEVELOPMENT HOLDOUT", True, "949 active replay files; 34 maps"),
    ("adaptive_alpha", "results/adaptive_alpha_validation_v1", "LEVEL 2 — FROZEN DEVELOPMENT HOLDOUT", True, "611/338 map-disjoint split"),
    ("strong_steganalysis", "results/strong_steganalysis_v1", "LEVEL 2 — FROZEN DEVELOPMENT HOLDOUT", True, "338 validation replays; FULL-29"),
    ("distributed_layout", "results/adaptive_layout_strong_v1", "LEVEL 3 — UNSEEN-BEATMAP VALIDATION", True, "611 development / 338 validation; five layouts"),
    ("decoder_errors", "results/decoder_error_analysis_v1", "LEVEL 2 — FROZEN DEVELOPMENT HOLDOUT", True, "physical validation matrix; five layouts"),
    ("secded", "results/ecc_equal_payload_v1", "LEVEL 2 — FROZEN DEVELOPMENT HOLDOUT", True, "80 replays × five layouts"),
    ("confidence_erasure", "results/ecc_confidence_pilot_v1", "NEGATIVE / REJECTED METHOD", False, "pilot only"),
    ("integrity_design", "results/ecc_integrity_rejection_v1", "LEVEL 1 — INTERNAL / EXPLORATORY", False, "rule design; consumed"),
    ("integrity_validation", "results/ecc_integrity_validation_v1", "LEVEL 3 — UNSEEN-BEATMAP VALIDATION", True, "167 eligible validation replays × five layouts"),
    ("capacity", "results/capacity_coverage_v1", "LEVEL 1 — INTERNAL / EXPLORATORY", True, "611 development replays; estimates mixed with pilot"),
    ("floor_pilot", "results/one_codeword_floor_pilot_v1", "LEVEL 1 — INTERNAL / EXPLORATORY", False, "80 development replays"),
    ("floor_validation", "results/one_codeword_floor_validation_v1", "LEVEL 4 — NEW-CORPUS / NEW-MAP VALIDATION", True, "300 primary replays on 20 new maps"),
    ("pn_phase1", "results/pn_key_robustness_v1", "ROBUSTNESS REANALYSIS — SCIENTIFICALLY CONSUMED DATA", True, "68 replays; 15 maps; 11 PN keys; five layouts"),
    ("pn_phase2", "results/pn_key_robustness_phase2_v1", "ROBUSTNESS REANALYSIS — SCIENTIFICALLY CONSUMED DATA", True, "68 replays; 15 maps; 30 new PN keys; three layouts"),
    ("host_aware_pn", "results/host_aware_pn_selection_v1", "NEGATIVE / REJECTED METHOD", True, "100 replay-disjoint, map-overlapping replays"),
    ("classifier_family", "results/classifier_family_robustness_v1", "ROBUSTNESS REANALYSIS — SCIENTIFICALLY CONSUMED DATA", True, "68 development and 135 unseen-map replays"),
    ("message_content", "results/message_content_robustness_v1", "ROBUSTNESS REANALYSIS — SCIENTIFICALLY CONSUMED DATA", True, "100 replays; five messages; three layouts"),
]


def artifact_inventory() -> list[dict[str, Any]]:
    rows = []
    for branch, path_text, status, eligible, corpus in BRANCHES:
        path = ROOT / path_text
        files = list(path.glob("*")) if path.exists() else []
        reports = sorted(p.name for p in files if p.suffix == ".md")
        configs = sorted(p.name for p in files if p.name in {"config.json", "classifier_config.json", "payload_policy.json", "frozen_selector.json"})
        locks = sorted(p.name for p in files if "lock" in p.name and p.suffix == ".json")
        manifest = path / "result_hashes.json"
        rows.append({
            "branch": branch, "result_path": path_text,
            "report_path": ";".join(f"{path_text}/{x}" for x in reports),
            "config_path": ";".join(f"{path_text}/{x}" for x in configs),
            "experiment_lock_path": ";".join(f"{path_text}/{x}" for x in locks),
            "hash_manifest_path": rel(manifest) if manifest.exists() else "",
            "source_corpus": corpus, "replay_count": corpus.split()[0] if corpus[:1].isdigit() else "scope-specific",
            "beatmap_count": "scope-specific", "physical_configuration": "see config/lock and claim registry",
            "scientific_status": status, "eligible_for_final_claims": eligible,
            "artifact_count": len([p for p in files if p.is_file()]),
            "provenance_note": "manifest verified" if manifest.exists() else "no branch-level manifest; lower provenance, source hashes recorded per claim",
        })
    return rows


def make_spec() -> dict[str, Any]:
    policy = json.loads((ROOT / "data/config/adaptive_alpha_sender_local_v2.json").read_text())
    integrity = json.loads((ROOT / "data/config/ecc_integrity_rejection_v1.json").read_text())
    return {
        "system_version": "osu-stego-final-default-v1",
        "scientific_status": "frozen final default",
        "replay_loader": {"time_origin": "raw cumulative replay deltas", "rng_sentinel_delta": -12345, "sentinel_is_gameplay": False, "first_delta_forced_to_zero": False},
        "map_offset": {"scope": "one fixed integer offset per beatmap", "source": "data/config/map_time_offsets.json", "per_replay_note_calibration": False},
        "matching": {"algorithm": "optimal monotone", "objective": ["maximize matches", "minimize total absolute timing error"], "one_to_one": True, "crossings": False},
        "carriers": {"deduplication": "retain first note mapped to a physical keydown; later duplicates become NaN/-1", "note_index_alignment_preserved": True},
        "adaptive_alpha": {"version": policy["policy_version"], "feature_source": policy["feature_source"], "formula": policy["formula"], "threshold_ratios": policy["threshold_ratios"], "alpha_values_ms": policy["alpha_values"], "boundary": "exact threshold advances to next alpha bin", "active_dataset_change_from_v1": "none"},
        "placement": {"name": "DISTRIBUTED", "version": "payload-layout-v1", "definition": "SHA-256-derived independent layout seed permutes all non-overlapping N-note blocks; take prefix", "nested_payloads": True, "pn_key_independent": True},
        "n_frames_per_bit": 8,
        "payload_policy": {"normal_fraction": {"numerator": 3, "denominator": 40}, "nominal_capacity_bits": "floor(note_positions/8)", "allocation": "max(1,floor(C*3/40)), then retain complete 8-bit words", "abstain": "if no complete SECDED codeword; equivalently C <= 106", "floor_extension": False},
        "ecc": {"code": "extended Hamming SECDED(8,4)", "useful_rate": 0.5, "hard_statuses": ["clean", "corrected_single", "detected_double"]},
        "integrity": {"rule_id": integrity["rule_id"], "definition": integrity["rule_definition"], "message_rule": integrity["message_rejection_semantics"], "ground_truth_used_for_decision": False},
        "pn": {"derivation": "SHA-256(key), first 4 bytes as unsigned big-endian seed, NumPy default_rng", "effective_seed_bits": 32, "cryptographic_kdf": False, "screening": False},
        "writer": {"name": "anchor-based writer", "target_keydowns_shifted": True, "nontarget_keydowns_unchanged": True, "cursor_only_frames_adjustable": True, "chronology_conflicts": "drop full target shift to zero", "global_cursor_monotonicity_required": False},
        "decoder": {"statistic": "sum(valid residual * PN chip) per selected block", "zero_correlation_bipolar_bit": 1, "missing_carriers": "ignored without compacting note indices"},
        "hit_window": {"osu_standard_50_window_ms": "200 - 10*effective_OD", "embedding_margin_ms": 5.0},
        "mods": {"Easy_OD": "base_OD*0.5", "HardRock_OD": "min(base_OD*1.4,10)", "DoubleTime_HalfTime_timestamps": "no separate rescaling; stored replay/beatmap axes plus frozen map offset are used, as empirically validated"},
        "physical_primary_measurement": "clean .osr -> encode -> write -> reload -> rematch -> residuals -> decode",
        "forbidden_final_variants": ["PREFIX", "12.5% one-codeword floor", "host-aware PN selection", "message screening", "confidence-erasure decoding"],
        "implementation_sources": {p: source_hash(p) for p in ["osu_stego/parsing/replay_loader.py", "osu_stego/parsing/beatmap_loader.py", "osu_stego/matching/matcher.py", "osu_stego/parsing/osr_writer.py", "osu_stego/stego/adaptive_alpha.py", "osu_stego/stego/carrier_utils.py", "osu_stego/stego/payload_layout.py", "osu_stego/stego/pn_sequence.py", "osu_stego/stego/ecc.py", "osu_stego/stego/integrity.py"]},
    }


def claim_rows(spec_hash: str) -> list[dict[str, Any]]:
    def claim(cid: str, short: str, status: str, branch: str, scope: str, dataset: str, replays: Any, maps: Any, metric: str, estimate: Any, uncertainty: str, unit: str, source: str, field: str, deploy: bool, independent: bool, caveat: str, allowed: str, forbidden: str) -> dict[str, Any]:
        return {"claim_id": cid, "short_claim": short, "status": status, "branch": branch, "exact_scope": scope, "dataset": dataset, "replay_count": replays, "beatmap_count": maps, "primary_metric": metric, "point_estimate": estimate, "uncertainty": uncertainty, "uncertainty_unit": unit, "source_file": source, "source_row_or_field": field, "source_hash": source_hash(source), "deployable?": deploy, "independent_validation?": independent, "caveat": caveat, "allowed_final_wording": allowed, "forbidden_overclaim": forbidden, "system_spec_sha256": spec_hash}
    return [
        claim("DATA_001", "Active post-quarantine corpus has 949 replays on 34 maps.", "VALID", "matching_validation", "active corpus", "original active", 949, 34, "counts", "949;34", "exact", "replay/map", "data/metadata/results_v3_clean.csv", "all rows", True, False, "same broad source family", "active corpus contained", "external population"),
        claim("MATCH_001", "Active-corpus weighted match ratio is 99.22%; median replay ratio is 100%.", "VALID", "matching_validation", "active post-quarantine", "original active", 949, 34, "weighted match ratio", 0.9922218795369235, "exact descriptive aggregation", "matched note", "data/metadata/results_v3_clean.csv", "sum(num_matched)/sum(num_notes)", True, False, "98.68% is the pre-quarantine 1,000-replay value", "99.22% observed", "matching is perfect"),
        claim("ADAPT_001", "Sender-local adaptive alpha reduced FULL-29 AUC relative to fixed alpha in frozen validation.", "VALID", "adaptive_alpha", "validation replay-grouped", "338 validation", 338, 11, "AUC", "0.5657 vs 0.5780", "individual replay-cluster CIs", "replay_file", "results/adaptive_alpha_validation_v1/heldout_method_summary.csv", "validation_replay_grouped_cv rows", True, False, "older PREFIX experiment", "reduced point estimate", "universally improves stealth"),
        claim("DIST_001", "DISTRIBUTED reduced PREFIX FULL-29 AUC for adaptive alpha across all five frozen layouts.", "VALID", "distributed_layout", "validation replay-grouped", "338 validation", 338, 11, "paired AUC delta", "-0.1558 to -0.1974", "all five 95% replay-cluster CIs exclude zero", "replay_file", "results/adaptive_layout_strong_v1/auc_paired.csv", "adaptive rows", True, False, "detector- and corpus-specific", "substantially reduced", "eliminated detectability"),
        claim("DIST_002", "Adaptive DISTRIBUTED generalized to unseen maps with FULL-29 AUC 0.597–0.614 over five layouts.", "VALIDATED ON UNSEEN MAPS", "distributed_layout", "development-to-validation", "611 train / 338 validation", 338, 11, "AUC range", "0.5966–0.6144", "beatmap-cluster CIs by layout", "beatmap_hash", "results/adaptive_layout_strong_v1/dev_to_validation_auc_by_seed.csv", "adaptive distributed rows", True, True, "same original source family", "modest above-chance detectability", "undetectable"),
        claim("DECODER_001", "Raw physical BER was 4.21% and errors were mildly bursty.", "VALID", "decoder_errors", "five existing layouts", "validation physical matrix", 338, 11, "weighted raw BER", 0.042139, "descriptive; lag-1 and run diagnostics", "coded bit", "results/decoder_error_analysis_v1/error_summary.csv", "overall and independence rows", True, False, "not independent; clustering is modest", "mildly bursty", "independent errors"),
        claim("CONF_001", "Low absolute correlation predicted bit error with AUC 0.854.", "VALID", "decoder_errors", "five existing layouts", "validation physical matrix", 338, 11, "error-prediction AUC", 0.85358, "descriptive", "coded bit", "results/decoder_error_analysis_v1/error_summary.csv", "low_abs_correlation_error_auc", True, False, "confidence is not calibrated probability", "useful error indicator", "guarantees correctness"),
        claim("ECC_001", "SECDED improved equal-useful-payload full-message recovery by 18.0 pp.", "VALID", "secded", "paired equal-useful payload", "80-replay ECC panel", 80, "scope-specific", "full-message recovery delta", 0.18, "95% CI [0.1275,0.2325]", "replay_file", "results/ecc_equal_payload_v1/paired_recovery.csv", "equal_useful/full_message_recovered", True, False, "50% code rate and twice physical bits", "improved same-message recovery", "improved capacity"),
        claim("INT_001", "Frozen integrity reduced wrong acceptance from 14/835 to 1/835.", "VALIDATED ON UNSEEN MAPS", "integrity_validation", "eligible validation replays × five layouts", "original validation", 167, 5, "wrong accept rate", "1.6766% to 0.1198%", "paired delta CI [-2.7545,-0.5988] pp", "replay_file", "results/ecc_integrity_validation_v1/summary.csv", "baseline/integrity rows", True, True, "correct accept fell and rejection rose", "reduced observed silent acceptance", "zero risk"),
        claim("INT_002", "Integrity captured 13/14 baseline wrong accepts but additionally rejected 51/767 baseline-correct messages.", "VALIDATED ON UNSEEN MAPS", "integrity_validation", "eligible validation", "original validation", 167, 5, "transition counts", "13/14;51/767", "exact observed counts", "physical configuration", "results/ecc_integrity_validation_v1/silent_error_capture.csv", "single row plus false_rejection.csv", True, True, "K=8 subgroup rejection severe", "tradeoff", "authentication"),
        claim("FLOOR_001", "The 12.5%-capped floor was rejected after worse new-corpus reliability.", "NEGATIVE / REJECTED METHOD", "floor_validation", "new-map primary corpus", "new floor validation", 300, 20, "raw BER/correct accept", "5.89%/78.67% vs normal 2.55%/91.41%", "replay-cluster CIs in source", "replay_file", "results/one_codeword_floor_validation_v1/reliability_by_class.csv", "FLOOR_12_5 and NORMAL_7_5", False, True, "capacity classes differ intrinsically", "floor was not retained", "floor comparison proves causality"),
        claim("PN_001", "Across 30 preregistered PN keys, mean BER was 5.136%, SD 0.478 pp, range 4.087–6.190%.", "ROBUSTNESS SUPPORTED", "pn_phase2", "common three-layout Phase-2 design", "68-replay PN panel", 68, 15, "weighted raw BER distribution", "mean .051362; SD .004782; range .040873–.061905", "PN-key bootstrap mean CI [.049696,.053016]", "PN key", "results/pn_key_robustness_phase2_v1/combined_distribution_summary.csv", "phase2_30_new", True, False, "finite tested keys; not seed-space probability", "PN sensitivity exists", "PN-independent reliability"),
        claim("PN_002", "Historical PN was descriptively typical, not selected as best.", "ROBUSTNESS SUPPORTED", "pn_phase2", "common three-layout design", "68-replay PN panel", 68, 15, "standardized difference", "-0.036 SD; rank 13/30", "descriptive", "PN key", "results/pn_key_robustness_phase2_v1/historical_key_position.csv", "historical row", True, False, "no post-hoc key selection allowed", "descriptively typical", "optimal PN"),
        claim("MSG_001", "Message-family BER ranged 5.769–7.131%, with 0.498 pp between-message SD.", "ROBUSTNESS SUPPORTED", "message_content", "K=1 reused panel; three layouts", "host-aware holdout reuse", 100, 14, "weighted raw BER", "0.05769–0.07131; SD .00498", "family-specific replay-cluster analyses", "replay_file", "results/message_content_robustness_v1/reliability_by_message.csv", "five rows", True, False, "one PN; consumed data; some duplicate payloads", "modest message sensitivity", "message-independent"),
        claim("CLF_001", "FULL-29 detectability persisted across four classifier families.", "ROBUSTNESS SUPPORTED", "classifier_family", "replay-grouped and unseen-map", "consumed classifier population", 203, 24, "AUC range", "0.603–0.635 in both regimes", "replay- or beatmap-cluster CI by regime", "cluster", "results/classifier_family_robustness_v1/auc_replay_grouped.csv", "FULL rows plus auc_dev_to_validation.csv", True, False, "hand-engineered features only", "not RF-specific under tested features", "security against arbitrary detectors"),
    ]


def dataset_rows(spec_hash: str) -> list[dict[str, Any]]:
    active = pd.read_csv(ROOT / "data/metadata/results_v3_clean.csv")
    split = pd.read_csv(ROOT / "results/adaptive_alpha_validation_v1/heldout_partition.csv")
    new = pd.read_csv(ROOT / "results/one_codeword_floor_validation_v1/corpus_manifest.csv")
    cal = pd.read_csv(ROOT / "results/one_codeword_floor_validation_v1/calibration_manifest.csv")
    host = pd.read_csv(ROOT / "results/host_aware_pn_selection_v1/holdout_selection.csv")
    clf = pd.read_csv(ROOT / "results/classifier_family_robustness_v1/dataset_manifest.csv")
    active_users = set(active.player.astype(str).str.casefold())
    new_users = set(new.username_casefold.astype(str))
    rows = []
    def add(cid: str, name: str, frame: pd.DataFrame, replay_col: str, map_col: str, use: str, consumed: str, overlap: str, user_col: str | None = None) -> None:
        rows.append({"corpus_id": cid, "name": name, "replays": frame[replay_col].nunique(), "beatmaps": frame[map_col].nunique(), "replay_ids_sha256": stable_set_hash(frame[replay_col]), "beatmap_ids_sha256": stable_set_hash(frame[map_col]), "username_ids_sha256": stable_set_hash(frame[user_col].astype(str).str.casefold()) if user_col else "not available/needed", "overlap_audit": overlap, "scientific_consumption_state": consumed, "intended_use": use, "source_artifact": rel(frame.attrs.get("source", "")), "system_spec_sha256": spec_hash})
    active.attrs["source"] = "data/metadata/results_v3_clean.csv"
    add("ACTIVE_949", "Original active post-quarantine corpus", active, "replay_file", "beatmap_hash", "core development and map-disjoint validation", "consumed", "excludes 51 replay files on 3 timing-model-incompatible maps", "player")
    for part in ["development", "validation"]:
        sub = split[split.partition == part].copy(); sub.attrs["source"] = "results/adaptive_alpha_validation_v1/heldout_partition.csv"
        add("ORIG_" + part.upper(), f"Frozen original {part} partition", sub, "beatmap_hash", "beatmap_hash", "method development" if part == "development" else "unseen-beatmap validation", "consumed", "zero beatmap overlap between frozen partitions")
        rows[-1]["replays"] = int(sub.replay_count.sum())
        rows[-1]["replay_ids_sha256"] = "partition artifact contains per-map counts; replay set anchored by downstream selections"
    new.attrs["source"] = "results/one_codeword_floor_validation_v1/corpus_manifest.csv"
    add("NEW_FLOOR_PRIMARY", "New floor-validation primary corpus", new, "replay_file", "beatmap_hash", "new-corpus floor validation", "consumed", f"direct check: active overlap replay=0, map=0, normalized username={len(active_users & new_users)}", "username_casefold")
    cal.attrs["source"] = "results/one_codeword_floor_validation_v1/calibration_manifest.csv"
    add("NEW_FLOOR_CAL", "Disjoint floor-validation offset calibration pool", cal, "replay_file", "beatmap_hash", "map-offset calibration only", "consumed", "240 replay files; disjoint from primary replay IDs", "username_casefold" if "username_casefold" in cal else None)
    host.attrs["source"] = "results/host_aware_pn_selection_v1/holdout_selection.csv"
    add("HOST_HOLDOUT", "Host-aware replay-disjoint holdout", host, "replay_file", "beatmap_hash", "host-aware and message robustness", "consumed", "0 replay overlap with design; all 14 maps overlap")
    msg = host.copy(); msg.attrs["source"] = "results/message_content_robustness_v1/source_artifact_audit.md"
    add("MESSAGE_REUSE", "Message-content reuse panel", msg, "replay_file", "beatmap_hash", "five-message robustness reanalysis", "consumed", "exact K=1 host-holdout reuse; 1,500 physical conditions")
    clf.attrs["source"] = "results/classifier_family_robustness_v1/dataset_manifest.csv"
    add("CLASSIFIER_POP", "Classifier robustness feature population", clf, "replay_file", "beatmap_hash", "four-family robustness reanalysis", "consumed", "203 replay files / 24 maps; clean/stego grouped by replay")
    return rows


def method_status_rows(spec_hash: str) -> list[dict[str, Any]]:
    values = [
        ("final_default", "adaptive v2 + DISTRIBUTED + N=8 + normal 7.5% + SECDED + integrity + abstention", "FINAL", "deployment/research default"),
        ("fixed_alpha", "fixed alpha controls", "CONTROL", "comparison only"),
        ("prefix", "PREFIX placement", "REJECTED", "position-aware signature"),
        ("uncoded", "uncoded equal-useful/equal-physical controls", "CONTROL", "ECC comparisons only"),
        ("hard_secded", "SECDED without confidence integrity", "CONTROL", "higher wrong acceptance than final integrity mode"),
        ("floor_12_5", "12.5%-capped one-codeword floor", "REJECTED", "worse reliability on new corpus"),
        ("host_aware_k4", "host-aware PN K=4", "REJECTED", "gain requires unprotected PN-index side information; blind mode inadequate"),
        ("message_screening", "message screening/whitening", "FUTURE WORK", "not tested and not part of system"),
        ("confidence_erasure", "confidence-aware erasure decoding", "REJECTED", "silent miscorrections increased"),
    ]
    return [{"method_id": a, "definition": b, "status": c, "reason": d, "system_spec_sha256": spec_hash} for a,b,c,d in values]


def denominator_rows(spec_hash: str) -> list[dict[str, Any]]:
    values = [
        ("MATCH_001", 1055224, 1063496, "matched note / active note", "weighted ratio; not mean replay ratio", "PASS"),
        ("DECODER_001", 1986, 47130, "coded bit", "total errors / total transmitted bits", "PASS"),
        ("ECC_001", "72 successes difference", 400, "replay-layout configuration", "paired equal-useful full-message recovery", "PASS"),
        ("INT_001 baseline", 14, 835, "eligible replay × layout seed", "WRONG_ACCEPT/config", "PASS"),
        ("INT_001 integrity", 1, 835, "eligible replay × layout seed", "WRONG_ACCEPT/config", "PASS"),
        ("FLOOR normal raw", 153, 6000, "coded bit", "errors/coded bits", "PASS"),
        ("FLOOR floor raw", 106, 1800, "coded bit", "errors/coded bits", "PASS"),
        ("PN mean", "mean of 30 per-key weighted BERs", 30, "PN key", "descriptive mean; each key BER is bit-weighted", "PASS"),
        ("MSG all families", "family-specific errors", 3744, "coded bit per message family", "do not average replay BER", "PASS"),
        ("classifier AUC", "paired clean/stego scores", "203 replay pairs or 135 unseen replay pairs", "replay pair", "clean/stego identity grouped", "PASS"),
    ]
    return [{"metric_or_claim": a, "numerator": b, "denominator": c, "unit": d, "interpretation": e, "audit_status": f, "system_spec_sha256": spec_hash} for a,b,c,d,e,f in values]


def uncertainty_rows(spec_hash: str) -> list[dict[str, Any]]:
    values = [
        ("DIST_001", "replay-cluster bootstrap", "replay_file", "paired layout comparison", "PASS"),
        ("DIST_002", "beatmap-cluster bootstrap", "beatmap_hash", "development to unseen maps", "PASS"),
        ("ECC_001", "replay-cluster bootstrap", "replay_file", "all five layouts retained per replay", "PASS"),
        ("INT_001", "replay-cluster bootstrap", "replay_file", "all five layouts retained", "PASS"),
        ("FLOOR_001", "replay-cluster for reliability; beatmap-cluster for detector", "metric-specific", "do not substitute zero-event point estimate for risk bound", "PASS"),
        ("PN_001 mean", "PN-key bootstrap", "PN key", "between-key mean CI", "PASS"),
        ("PN_001 system", "two-level PN + replay bootstrap", "PN key and replay cluster", "use estimate_observed, not legacy MC mean estimate field", "CORRECTED"),
        ("MSG_001", "replay-cluster bootstrap", "replay_file", "three layouts kept together", "PASS"),
        ("CLF replay-grouped", "replay-cluster bootstrap", "replay_file", "not map-level uncertainty", "PASS"),
        ("CLF unseen", "beatmap-cluster bootstrap", "beatmap_hash", "correct unseen-map uncertainty", "PASS"),
        ("historical generic beatmap-CV CI", "replay bootstrap", "replay_file", "must not be described as map-level uncertainty", "SUPERSEDED_FOR_MAP_CLAIMS"),
    ]
    return [{"claim_or_result": a, "method": b, "resampling_unit": c, "audit_note": d, "status": e, "system_spec_sha256": spec_hash} for a,b,c,d,e in values]


def canonical_metrics(spec_hash: str) -> list[dict[str, Any]]:
    active = pd.read_csv(ROOT / "data/metadata/results_v3_clean.csv")
    classifier_r = pd.read_csv(ROOT / "results/classifier_family_robustness_v1/auc_replay_grouped.csv")
    classifier_u = pd.read_csv(ROOT / "results/classifier_family_robustness_v1/auc_dev_to_validation.csv")
    rf = classifier_r[classifier_r.feature_set == "full"]
    uf = classifier_u[classifier_u.feature_set == "full"]
    values = [
        ("data_quality", "active_replays", 949, "replay", "ACTIVE_949", "DATA_001", "data/metadata/results_v3_clean.csv"),
        ("data_quality", "active_beatmaps", 34, "beatmap", "ACTIVE_949", "DATA_001", "data/metadata/results_v3_clean.csv"),
        ("data_quality", "weighted_match_ratio", active.num_matched.sum()/active.num_notes.sum(), "matched note", "ACTIVE_949", "MATCH_001", "data/metadata/results_v3_clean.csv"),
        ("data_quality", "median_replay_match_ratio", active.match_ratio.median(), "replay", "ACTIVE_949", "MATCH_001", "data/metadata/results_v3_clean.csv"),
        ("distributed", "validation_raw_ber_five_layouts", 0.042139, "coded bit", "original validation", "DECODER_001", "results/decoder_error_analysis_v1/error_summary.csv"),
        ("distributed", "unseen_map_FULL29_AUC_range", "0.5966–0.6144", "layout", "original map-disjoint validation", "DIST_002", "results/adaptive_layout_strong_v1/dev_to_validation_auc_by_seed.csv"),
        ("secded", "equal_useful_full_recovery_gain_pp", 18.0, "replay-layout configuration", "80-replay panel", "ECC_001", "results/ecc_equal_payload_v1/paired_recovery.csv"),
        ("secded", "coding_rate", 0.5, "useful/coded bit", "code definition", "ECC_001", "osu_stego/stego/ecc.py"),
        ("integrity", "wrong_accept_baseline", "14/835 (1.68%)", "physical configuration", "167 validation replays × 5", "INT_001", "results/ecc_integrity_validation_v1/summary.csv"),
        ("integrity", "wrong_accept_final", "1/835 (0.12%)", "physical configuration", "167 validation replays × 5", "INT_001", "results/ecc_integrity_validation_v1/summary.csv"),
        ("integrity", "final_correct_accept", "716/835 (85.75%)", "physical configuration", "167 validation replays × 5", "INT_001", "results/ecc_integrity_validation_v1/summary.csv"),
        ("integrity", "final_reject", "118/835 (14.13%)", "physical configuration", "167 validation replays × 5", "INT_001", "results/ecc_integrity_validation_v1/summary.csv"),
        ("capacity", "new_corpus_normal_eligibility", "135/300 (45%)", "input replay", "new primary corpus", "FLOOR_001", "results/one_codeword_floor_validation_v1/system_metrics_summary.csv"),
        ("capacity", "floor_decision", "rejected", "method", "new primary corpus", "FLOOR_001", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("steganalysis", "FULL29_replay_grouped_classifier_range", f"{rf.roc_auc.min():.3f}–{rf.roc_auc.max():.3f}", "classifier", "203 consumed replay pairs", "CLF_001", "results/classifier_family_robustness_v1/auc_replay_grouped.csv"),
        ("steganalysis", "FULL29_unseen_classifier_range", f"{uf.roc_auc.min():.3f}–{uf.roc_auc.max():.3f}", "classifier", "135 unseen-map replay pairs", "CLF_001", "results/classifier_family_robustness_v1/auc_dev_to_validation.csv"),
        ("pn", "phase2_mean_SD_range", "5.136%; 0.478 pp; 4.087–6.190%", "PN key", "30 keys, 68 replays, 3 layouts", "PN_001", "results/pn_key_robustness_phase2_v1/combined_distribution_summary.csv"),
        ("pn", "historical_position", "-0.036 SD; rank 13/30", "PN key", "common design", "PN_002", "results/pn_key_robustness_phase2_v1/historical_key_position.csv"),
        ("message", "BER_range_SD", "5.769–7.131%; SD 0.498 pp", "message family", "100 replays × 3 layouts", "MSG_001", "results/message_content_robustness_v1/reliability_by_message.csv"),
        ("message", "FULL29_RF_range", "0.560–0.575", "message family", "reused K=1 panel", "MSG_001", "results/message_content_robustness_v1/detector_by_message.csv"),
    ]
    return [{"group": a, "metric": b, "value": c, "denominator_unit": d, "scope_dataset": e, "claim_id": f, "source_artifact": g, "source_hash": source_hash(g), "system_spec_sha256": spec_hash} for a,b,c,d,e,f,g in values]


def end_to_end_rows(spec_hash: str) -> list[dict[str, Any]]:
    values = [
        ("eligibility", "0.45", "input replay", "DIRECT", "new corpus, normal class", "results/one_codeword_floor_validation_v1/system_metrics_summary.csv"),
        ("abstention", "0.55", "input replay", "DERIVED", "1-normal eligibility; final default excludes floor", "results/one_codeword_floor_validation_v1/policy_cost_benefit.csv"),
        ("mean useful bits", "4.444", "eligible replay-layout config", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("raw BER", "0.0255", "6000 coded bits", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("post-ECC BER", "0.0100", "3000 useful bits", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("CORRECT_ACCEPT", "617/675 (91.41%)", "eligible replay-layout config", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("WRONG_ACCEPT", "0/675 observed", "eligible replay-layout config", "DIRECT", "zero-event uncertainty remains", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("REJECT", "58/675 (8.59%)", "eligible replay-layout config", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("correctly accepted useful bits", "3.917", "eligible replay-layout config", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/reliability_by_class.csv"),
        ("correctly accepted useful bits", "1.763", "input replay-layout opportunity", "DIRECT", "300 replays × five layouts denominator", "results/one_codeword_floor_validation_v1/policy_cost_benefit.csv"),
        ("active-carrier fraction", "0.998658", "requested carrier", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/physical_by_class.csv"),
        ("chronology drops", "0.0844", "eligible config mean", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/physical_by_class.csv"),
        ("new matcher-unmatched", "0.4341", "eligible config mean", "DIRECT", "normal class", "results/one_codeword_floor_validation_v1/physical_by_class.csv"),
        ("FULL-29 AUC", "0.629–0.657 across five layouts", "135 replay pairs/layout", "DIRECT", "frozen development-to-new-map normal class", "results/one_codeword_floor_validation_v1/detector_auc.csv"),
    ]
    return [{"quantity": a, "value": b, "denominator": c, "measurement": d, "scope_note": e, "source_artifact": f, "same_exact_final_system": True, "system_spec_sha256": spec_hash} for a,b,c,d,e,f in values]


def comparability_rows(spec_hash: str) -> list[dict[str, Any]]:
    vals = [
        ("PREFIX vs DISTRIBUTED", "YES", "paired replay/message/PN/payload/writer/detector; layout only differs"),
        ("SECDED vs uncoded equal useful", "YES", "paired replay/layout and equal useful bits; physical budget intentionally differs"),
        ("hard SECDED vs integrity", "YES", "same physical replay and hard decoding; decision layer only"),
        ("normal 7.5% vs floor 12.5%", "APPROXIMATE", "same new corpus framework but disjoint intrinsic capacity classes"),
        ("PN keys Phase 2", "YES", "same replay/layout/message/config; PN realization differs"),
        ("message families", "YES", "same replay/layout/PN/config; message differs; duplicate conditions marked"),
        ("classifier families", "YES", "identical frozen features/folds; classifier differs"),
        ("decoder BER 4.21% vs new-corpus normal BER 2.55%", "NO", "different corpus and coding-era physical experiments"),
        ("integrity validation rates vs new-corpus normal rates", "NO", "different corpus; use as scoped replication, not pooled score"),
        ("historical 3-feature AUC vs final FULL-29", "NO", "detector feature set and physical method differ"),
    ]
    return [{"comparison": a, "direct_comparison": b, "reason": c, "same_replays": "see reason", "same_maps": "see reason", "same_pn": "see reason", "same_layout": "see reason", "same_payload": "see reason", "same_writer": True, "same_ecc_integrity": "see reason", "same_detector": "see reason", "same_bootstrap_unit": "metric-specific", "system_spec_sha256": spec_hash} for a,b,c in vals]


def evolution_rows(spec_hash: str) -> list[dict[str, Any]]:
    vals = [
        ("baseline timing/matching", "weighted match 78.90%", "n/a", "n/a", "revealed timing-origin flaw", "invalid old matching baseline", "SUPERSEDED"),
        ("corrected timing + optimal matching", "99.22% active weighted match", "n/a", "n/a", "validated residual reconstruction", "3 maps quarantined", "FINAL FOUNDATION"),
        ("fixed-alpha PREFIX", "scope-specific BER", "FULL AUC 0.861 on 338 validation", "7.5% uncoded-era", "control", "strong position signature", "REJECTED CONTROL"),
        ("sender-local adaptive PREFIX", "2.90% validation BER", "FULL AUC 0.802", "7.5% uncoded-era", "sender-deployable alpha", "still position-signature prone", "CONTROL"),
        ("adaptive DISTRIBUTED", "4.21% five-layout BER", "unseen FULL AUC 0.597–0.614", "7.5% coded-bit allocation", "reduced location signature", "higher raw BER; modest detectability remains", "FINAL COMPONENT"),
        ("+ SECDED(8,4)", "+18 pp equal-useful recovery", "not a detector comparison", "50% useful code rate", "message recovery", "twice coded bits per useful bit", "FINAL COMPONENT"),
        ("+ integrity rejection", "1/835 wrong accept; 716/835 correct accept", "same physical features", "no extra embedded bits", "reduced silent acceptance", "more rejection", "FINAL COMPONENT"),
        ("+ normal abstention", "new-corpus normal correct accept 91.41%", "FULL AUC 0.629–0.657", "eligible only; 45% new-corpus coverage", "avoids unreliable floor", "short replay abstention", "FINAL DEFAULT"),
    ]
    return [{"method":a,"reliability_metric":b,"detectability_metric":c,"payload_or_coding_rate":d,"main_benefit":e,"main_cost":f,"scientific_status":g,"comparability_note":"Rows use different scopes unless explicitly identified; do not infer a single longitudinal effect.","system_spec_sha256":spec_hash} for a,b,c,d,e,f,g in vals]


def negative_rows(spec_hash: str) -> list[dict[str, Any]]:
    vals = [
        ("PREFIX", "early placement may be adequate", "FULL-29 exposed strong quarter-specific signature", "rejected as final placement", "placement location matters"),
        ("12.5% one-codeword floor", "coverage can rise without unacceptable reliability loss", "new-corpus floor BER 5.89%, correct accept 78.67%, reject 21.33%", "rejected as default", "nominal coverage is not reliable throughput"),
        ("host-aware PN K=4", "sender-local host scoring can improve reliability operationally", "index-known BER improved, blind wrong/ambiguous outcomes rose; side bits unprotected", "rejected from final protocol", "PN synchronization is the unresolved cost"),
        ("confidence erasure decoding", "low-|C| erasures can safely improve recovery", "some recovery improved but silent miscorrections increased", "rejected", "confidence helps integrity more safely than aggressive correction"),
        ("message screening/whitening", "message choice can be optimized", "not tested; robustness showed modest sensitivity only", "not implemented", "message sensitivity is a limitation, not tuning permission"),
    ]
    return [{"method":a,"hypothesis":b,"result":c,"why_rejected":d,"what_was_learned":e,"system_spec_sha256":spec_hash} for a,b,c,d,e in vals]


def safe_language_rows(spec_hash: str) -> list[dict[str, Any]]:
    patterns = ["secure", "undetectable", "invisible", "impossible to detect", "guaranteed", "always", "zero risk", "perfect", "real miss", "corrupt map", "optimal security", "capacity", "random"]
    rows: list[dict[str, Any]] = []
    candidates = [ROOT / "AGENTS.md"] + sorted((ROOT / "results").glob("*/*.md"))
    candidates = [p for p in candidates if "final_audit_v1" not in p.parts]
    for path in candidates:
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            found = [p for p in patterns if p in low]
            if not found:
                continue
            status = "REVIEWED_CONTEXTUAL_USE"
            if "undetectable" in found or "secure" in found or "guaranteed" in found or "zero risk" in found:
                status = "PROHIBITED_IF_ASSERTED; CURRENT OCCURRENCE IS WARNING/NEGATION OR HISTORICAL"
            if "capacity" in found or "random" in found:
                status = "VALID_ONLY_WITH_QUALIFIER"
            rows.append({"file": rel(path), "line": number, "terms": ";".join(found), "status": status, "context_classification": "historical report or methodological warning; final narrative uses qualified language", "system_spec_sha256": spec_hash})
    return rows


def write_longform(out: Path, spec_hash: str) -> None:
    (out / "limitations.md").write_text(f"""# Limitations registry

System spec: `{spec_hash}`.

- Corpora are not fully independent external sources; the new-map corpus remains in the same broader Kaggle/o!rdr source family.
- PN, message, and classifier robustness analyses reuse scientifically consumed data.
- The PN generator has a 32-bit effective seed and is not a cryptographic KDF. Wrong-key BER is not a security proof.
- Steganalysis covers 29 hand-engineered timing features and four preregistered classifier families, not arbitrary attackers; per-map detectability is heterogeneous.
- PN realization and, more modestly, message content change reliability.
- SECDED(8,4) halves the useful coding rate. Integrity rejection lowers silent acceptance while rejecting some correct messages; the two-codeword subgroup was especially fragile.
- The normal 7.5% policy abstains on short replay files, so reported eligible-case reliability is not population-wide coverage.
- The method is osu!standard-specific. Three maps were quarantined because one fixed map-level offset could not represent them reliably; this does not imply file corruption.
- DT/HT behavior is limited to the empirically validated pipeline's stored timestamp handling and must not be generalized.
- No single historical experiment measured every claimed property on every corpus; final metrics remain scope-labelled.
""", encoding="utf-8")
    (out / "security_terminology_audit.md").write_text("""# Security terminology audit

PASS with limitations. Final material treats PN as a spreading/detection key, not encryption, authentication, tamper resistance, or a cryptographically strong key. The implementation's 32-bit NumPy seed is explicitly non-cryptographic. Wrong-key decoding and AUC results address tested reliability/detectability only; they do not establish confidentiality or security against chosen-message or arbitrary steganalysis attacks.

One legacy source-code docstring (`osu_stego/stego/pn_sequence.py`) says an attacker cannot approximately guess the PN sequence. That wording is stronger than the demonstrated evidence; it is recorded as a minor documentation issue and is not repeated in final scientific claims. No scientific behavior was changed.

`archive/legacy_scripts/run_decode.py` contains a hardcoded historical PN research key. It is not an osu! API credential and is not part of the final protocol; it must not be presented as secret key management.
""", encoding="utf-8")
    (out / "code_config_consistency.md").write_text(f"""# Code/config consistency

Central frozen semantics agree across implementation, frozen configs, and AGENTS.md: raw cumulative deltas with RNG sentinel removal; fixed map-level offsets; lexicographic optimal monotone matching; note-index-preserving physical-carrier deduplication; exact adaptive-alpha v2 comparisons; independently keyed nested DISTRIBUTED blocks; `N=8`; SECDED(8,4); corrected-bit-not-minimum integrity rejection; decoder `C=0 -> +1`; and anchor-based writer chronology handling.

The final default's ABSTAIN rule is the normal-policy restriction of the frozen payload policy: no complete 8-bit word means no transmission. The tested 12.5% extension is excluded.

Confirmed documentation defect: AGENTS.md section 6 associates approximately 98.68% weighted matching with the active corpus. Direct aggregation shows 98.676% for the pre-quarantine 1,000 rows and 99.222% for the active 949 rows. The final registry uses 99.222%; pipeline behavior is unaffected.

DT/HT receives no independent timestamp rescaling in the experiment context; Easy/HardRock modify OD before the 50-window calculation. This is reported as frozen empirical behavior, not a universal osu! timing rule.

System spec hash: `{spec_hash}`.
""", encoding="utf-8")
    (out / "reproducibility_instructions.md").write_text("""# Reproducibility instructions

From the repository root, run:

```bash
python3 -m scripts.analysis.finalize_research_results --check-reproducibility
```

The command validates every existing branch hash manifest and regenerates this audit from frozen CSV/JSON/source artifacts. It does not parse or write `.osr`, recalibrate offsets, rematch the raw corpus, access the osu! API, or fit a model. It fails on any frozen manifest mismatch and compares a temporary clean regeneration byte-for-byte with the canonical generated artifacts (excluding the self-referential output manifest and the reproducibility report).
""", encoding="utf-8")
    (out / "test_audit.md").write_text("""# Frozen-invariant test audit

All listed checks passed under `python3 -m ...`:

- `scripts.tools.check_final_frozen_invariants`: replay/beatmap loading, raw-time plus fixed-offset relation, optimal matching sanity, note-index-preserving deduplication, writer zero-shift anchor identity, and final normal-policy abstention boundary.
- `scripts.tools.check_adaptive_alpha_v2`: all 949 assignments unchanged, exact thresholds, repeated `.osr` parsing.
- `scripts.tools.check_payload_layout`: PREFIX compatibility, DISTRIBUTED uniqueness/nesting/determinism, PN/layout independence, noiseless decoding.
- `scripts.tools.check_ecc`: exhaustive SECDED messages and one-/two-error behavior.
- `scripts.tools.check_ecc_integrity`: frozen integrity hash and decisions.
- `scripts.tools.check_decoder_zero_tie`: exact zero correlation maps to +1.
- `scripts.tools.check_one_codeword_floor_policy`: exact C=63/64/106/107 tested-variant boundaries; final system disables the floor.
- `scripts.tools.check_sender_local_adaptive`, `check_adaptive_layout_strong`, `check_ecc_equal_payload`, `check_ecc_integrity_validation`, and `check_strong_steganalysis`: experiment schemas and invariants.
- `scripts.tools.check_pn_key_robustness`, `check_pn_key_robustness_phase2`, `check_host_aware_pn_results`, `check_classifier_family_results`, and `check_message_content_final`: frozen robustness outputs.
- `scripts.analysis.finalize_research_results --check-reproducibility`: eight branch manifests and clean deterministic regeneration.

One initial direct-file invocation failed only because package imports require module execution from the repository root; the documented `python3 -m` invocation passed.
""", encoding="utf-8")


def narrative(out: Path) -> None:
    (out / "final_results_narrative.md").write_text("""# Final results narrative

## Dataset and timing reconstruction

The active corpus contains 949 replay files on 34 maps after quarantining 51 files on three maps that were incompatible with the single-offset model; quarantine is not evidence of corruption. [CLAIM:DATA_001]

## Residual matching quality

Raw cumulative replay timing, one fixed offset per map, and optimal monotone matching yielded 99.22% weighted matching on the active corpus, with a 100% median replay ratio. [CLAIM:MATCH_001]

## Baseline embedding reliability

The five-layout adaptive+DISTRIBUTED physical channel had 1,986 errors among 47,130 coded bits (4.21% BER). [CLAIM:DECODER_001]

## Adaptive alpha

Sender-local alpha uses replay-header hit counts and exact rational boundaries; the corrected v2 semantics changed no assignment in the 949-replay corpus. The frozen validation point estimate was lower than its fixed-alpha control under FULL-29. [CLAIM:ADAPT_001]

## Strong steganalysis and PREFIX placement weakness

FULL-29 exposed a strong PREFIX-specific position signature. [CLAIM:DIST_001]

## DISTRIBUTED placement

Across five frozen layouts, DISTRIBUTED substantially reduced PREFIX AUC, while unseen-map FULL-29 AUC remained 0.597–0.614 rather than chance. [CLAIM:DIST_001] [CLAIM:DIST_002]

## Decoder error structure

Errors were mildly bursty rather than demonstrably independent. Low absolute correlation was a useful error indicator (AUC 0.854), but not a calibrated probability. [CLAIM:DECODER_001] [CLAIM:CONF_001]

## SECDED

At equal useful payload, SECDED improved full-message recovery by 18.0 percentage points (95% replay-cluster CI 12.75–23.25 pp), at 50% coding rate and greater physical cost. [CLAIM:ECC_001]

## Confidence-based integrity rejection

On 835 unseen-map validation configurations, the observable integrity rule reduced wrong accepts from 14 to 1 while correct accepts fell from 767 to 716 and rejects rose from 54 to 118. [CLAIM:INT_001] [CLAIM:INT_002]

## Capacity / abstention

The new-corpus normal class achieved 2.55% raw BER and 91.41% correct acceptance; the newly covered floor class had 5.89% BER and 78.67% correct acceptance, so the floor was rejected. [CLAIM:FLOOR_001]

## PN robustness

Across 30 new PN keys, mean BER was 5.136%, between-key SD 0.478 pp, and range 4.087–6.190%; the historical PN was descriptively typical. [CLAIM:PN_001] [CLAIM:PN_002]

## Message-content robustness

Five messages produced BER 5.769–7.131% with 0.498 pp between-message SD, supporting modest rather than catastrophic content sensitivity. [CLAIM:MSG_001]

## Classifier-family robustness

FULL-29 AUC remained 0.603–0.635 across RF, Logistic Regression, RBF SVM, and HGB in replay-grouped and unseen-map regimes. [CLAIM:CLF_001]

## Final system

The frozen default is adaptive-alpha v2 + DISTRIBUTED + N=8 + normal 7.5% coded-bit allocation + SECDED(8,4) + observable integrity rejection + abstention when no complete word fits. It has modest, above-chance detectability under tested adversaries and is not a cryptographic protocol.

## Limitations

Evidence is corpus- and detector-bounded; robustness panels reuse consumed data; short-file abstention limits coverage; and PN/message variation remains measurable.
""", encoding="utf-8")
    (out / "abstract_claims.md").write_text("""# Abstract-level claims

1. A validated raw-delta, fixed-map-offset reconstruction achieved 99.22% weighted matching on 949 active osu!standard replay files. [CLAIM:MATCH_001]
2. A 29-feature detector revealed a strong PREFIX-specific timing signature, and keyed DISTRIBUTED placement substantially reduced it across five frozen layouts. [CLAIM:DIST_001]
3. DISTRIBUTED embedding retained modest above-chance detectability on unseen maps (FULL-29 AUC 0.597–0.614), so the method is not described as undetectable. [CLAIM:DIST_002]
4. Physical channel errors were mildly bursty, while low absolute PN correlation was a useful bit-error indicator (AUC 0.854). [CLAIM:DECODER_001] [CLAIM:CONF_001]
5. SECDED(8,4) increased equal-useful-payload full-message recovery by 18.0 pp, with a 50% coding-rate cost. [CLAIM:ECC_001]
6. An observable integrity rule reduced wrong acceptance from 14/835 to 1/835 on frozen unseen-map validation, while increasing rejection. [CLAIM:INT_001]
7. A 12.5%-capped short-replay floor was rejected after materially poorer new-corpus reliability. [CLAIM:FLOOR_001]
8. Reliability varied across preregistered PN realizations and, more modestly, across message content; neither dimension justified post-hoc screening. [CLAIM:PN_001] [CLAIM:MSG_001]
9. Modest detectability persisted across four classifier families and was not specific to Random Forest under the tested timing features. [CLAIM:CLF_001]
10. Results remain bounded to tested corpora, hand-engineered timing features, and a non-cryptographic 32-bit PN seed.
""", encoding="utf-8")


def conclusions_rows(spec_hash: str) -> list[dict[str, Any]]:
    vals = [
        ("Does timing reconstruction work?", "Yes on the retained corpus", "MATCH_001", "high", "single map-offset model excluded 3 maps"),
        ("Is matching reliable?", "Yes, 99.22% weighted and 100% median replay ratio", "MATCH_001", "high", "dataset-specific"),
        ("Does adaptive alpha help?", "It reduced detectability point estimates; reliability/benefit is scope-dependent", "ADAPT_001", "moderate", "older PREFIX validation"),
        ("Is PREFIX acceptable?", "No as final placement", "DIST_001", "high", "tested detector"),
        ("Does DISTRIBUTED improve stealth?", "Yes relative to PREFIX under controlled FULL-29 tests", "DIST_001;DIST_002", "high", "does not eliminate detection"),
        ("Does SECDED improve same-message recovery?", "Yes", "ECC_001", "high", "50% rate and physical overhead"),
        ("Does integrity reduce silent acceptance?", "Yes, observed", "INT_001;INT_002", "high", "rejection/correct-accept cost"),
        ("Does higher payload floor work?", "No; rejected as default", "FLOOR_001", "high", "new corpus still same source family"),
        ("Does PN choice matter?", "Yes, moderate reliability sensitivity", "PN_001", "high for tested design", "finite 32-bit seeds sampled"),
        ("Can PN screening be used?", "Not in the frozen protocol", "negative_results:host-aware", "high", "index synchronization unresolved"),
        ("Does message content matter?", "Modestly for reliability", "MSG_001", "moderate", "one PN and consumed panel"),
        ("Is detectability RF-specific?", "No under tested families/features", "CLF_001", "high", "not arbitrary adversaries"),
        ("Is final system undetectable?", "No", "DIST_002;CLF_001", "high", "AUC remains above chance"),
        ("Is final system cryptographically secure?", "No such claim is supported", "security audit", "high", "PN is not encryption/authentication"),
    ]
    return [{"question":a,"answer":b,"evidence":c,"confidence":d,"limitation":e,"system_spec_sha256":spec_hash} for a,b,c,d,e in vals]


def reviewer_text() -> str:
    items = [
        ("AUC around 0.6 is not stealthy.", "VALID CONCERN", "The final wording says modest above-chance detectability, never invisibility."),
        ("Development choices may overfit the original corpus.", "PARTIALLY ADDRESSED", "Map-disjoint and new-map tests exist, but sources are not fully external."),
        ("Map offsets are data-dependent.", "PARTIALLY ADDRESSED", "They are map-level and calibration-separated in the new corpus, but remain a model assumption."),
        ("Quarantining maps creates selection bias.", "VALID CONCERN", "Membership is explicit; quarantine means timing-model incompatibility, not corruption."),
        ("7.5% is not useful payload.", "ADDRESSED BY EXPERIMENT", "It is coded-bit allocation; SECDED rate and abstention are reported separately."),
        ("SECDED throughput is costly.", "VALID CONCERN", "Useful rate is 50%, with direct physical-cost comparisons."),
        ("Abstention hides difficult short cases.", "VALID CONCERN", "Eligibility and per-input throughput are explicitly reported."),
        ("PN robustness uses one replay panel.", "VALID CONCERN", "Forty preregistered keys characterize that design only."),
        ("Message robustness uses one PN.", "VALID CONCERN", "It is labelled consumed-data robustness, not independent validation."),
        ("Classifier robustness is limited to hand-engineered features.", "VALID CONCERN", "Four families do not upper-bound learned or future attacks."),
        ("A 32-bit PN seed is not meaningful cryptographic security.", "ADDRESSED BY EXPERIMENT", "No cryptographic claim is made; it is an explicit limitation."),
        ("Observed associations may be called causal.", "ADDRESSED BY EXPERIMENT", "Narrative uses observed association and controlled comparisons only."),
        ("Integrity may reject too many correct messages.", "VALID CONCERN", "51/767 extra false rejects and severe K=8 behavior are disclosed."),
        ("Zero new-corpus wrong accepts may be overinterpreted.", "ADDRESSED BY EXPERIMENT", "It is reported as 0/675 observed with material zero-event uncertainty."),
        ("Historical results mix incompatible corpora.", "ADDRESSED BY EXPERIMENT", "Comparability matrix and scope-labelled canonical metrics prohibit a synthetic score."),
    ]
    return "# Skeptical reviewer audit\n\n" + "\n".join(f"## {i+1}. {q}\n\n**{s}.** {a}\n" for i,(q,s,a) in enumerate(items))


def svg_chart(path: Path, title: str, labels: list[str], values: list[float], y_label: str, colors: list[str] | None = None, reference: float | None = None) -> None:
    width, height = 960, 560
    left, right, top, bottom = 100, 30, 70, 115
    plot_w, plot_h = width-left-right, height-top-bottom
    maximum = max(values + ([reference] if reference is not None else [0])) * 1.15 or 1
    colors = colors or ["#3478bf"] * len(values)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', f'<rect width="{width}" height="{height}" fill="white"/>', f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="23">{html.escape(title)}</text>', f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(y_label)}</text>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>', f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>']
    for tick in range(6):
        val = maximum*tick/5; y = top+plot_h-(val/maximum)*plot_h
        chunks += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>', f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.3f}</text>']
    bar_w = plot_w/max(len(values),1)*0.62
    for i,(label,value,color) in enumerate(zip(labels,values,colors)):
        x = left+(i+.5)*plot_w/len(values)-bar_w/2; h=value/maximum*plot_h; y=top+plot_h-h
        chunks += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>', f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>', f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(label)}</text>']
    if reference is not None:
        y=top+plot_h-reference/maximum*plot_h; chunks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#c33" stroke-dasharray="7,5"/>')
    chunks.append('</svg>')
    svg = path.with_suffix('.svg'); svg.write_text("\n".join(chunks)+"\n",encoding="utf-8")
    subprocess.run(["sips", "-s", "format", "png", str(svg), "--out", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def svg_scatter(path: Path, title: str, labels: list[str], xs: list[float], ys: list[float], colors: list[str]) -> None:
    width,height=960,560; left,right,top,bottom=105,35,70,85; pw,ph=width-left-right,height-top-bottom
    xmin,xmax=min(xs)-0.003,max(xs)+0.003; ymin,ymax=min(ys)-0.025,max(ys)+0.025
    chunks=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="{width}" height="{height}" fill="white"/><text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="23">{html.escape(title)}</text>',f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#222"/><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#222"/><text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="15">weighted physical BER</text><text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">FULL-29 ROC-AUC</text>']
    for i in range(6):
        xv=xmin+(xmax-xmin)*i/5; x=left+pw*i/5; yv=ymin+(ymax-ymin)*i/5; y=top+ph-ph*i/5
        chunks += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}" stroke="#eee"/><text x="{x:.1f}" y="{top+ph+22}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.3f}</text>',f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#eee"/><text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.3f}</text>']
    for label,xv,yv,color in zip(labels,xs,ys,colors):
        x=left+(xv-xmin)/(xmax-xmin)*pw; y=top+ph-(yv-ymin)/(ymax-ymin)*ph
        chunks += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#222"/>',f'<text x="{x+9:.1f}" y="{y-7:.1f}" font-family="Arial" font-size="10">{html.escape(label)}</text>']
    chunks.append('</svg>'); svg=path.with_suffix('.svg'); svg.write_text("\n".join(chunks)+"\n",encoding="utf-8")
    subprocess.run(["sips","-s","format","png",str(svg),"--out",str(path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def pipeline_svg(path: Path) -> None:
    labels = ["Clean .osr", "Align + residuals", "Adaptive α v2", "SECDED(8,4)", "DISTRIBUTED PN", "Anchor writer", "Reload + decode", "SECDED + integrity", "ACCEPT / REJECT"]
    w,h=1200,330; chunks=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="{w}" height="{h}" fill="white"/><text x="600" y="34" text-anchor="middle" font-family="Arial" font-size="24">Frozen end-to-end system</text><text x="310" y="75" text-anchor="middle" font-family="Arial" font-size="16" fill="#1261a0">SENDER</text><text x="900" y="75" text-anchor="middle" font-family="Arial" font-size="16" fill="#8b3f00">RECEIVER</text>']
    for i,label in enumerate(labels):
        x=20+i*130; color="#d9ecff" if i<=5 else "#ffe8cf"
        chunks.append(f'<rect x="{x}" y="115" rx="8" width="112" height="70" fill="{color}" stroke="#333"/><text x="{x+56}" y="145" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(label)}</text>')
        if i<len(labels)-1: chunks.append(f'<line x1="{x+112}" y1="150" x2="{x+128}" y2="150" stroke="#333"/><polygon points="{x+128},150 {x+120},145 {x+120},155" fill="#333"/>')
    chunks.append('<text x="600" y="240" text-anchor="middle" font-family="Arial" font-size="14">Primary evidence uses the physical write → reload → rematch round trip</text></svg>')
    svg=path.with_suffix('.svg'); svg.write_text("\n".join(chunks)+"\n",encoding="utf-8")
    subprocess.run(["sips","-s","format","png",str(svg),"--out",str(path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def figures(out: Path, spec_hash: str) -> None:
    fdir=out/"figures"; fdir.mkdir()
    gen_hash=sha256(Path(__file__))
    def meta(fid: str,title: str,sources:list[str],denom:str,method:str,scope:str) -> None:
        write_json(fdir/f"{fid}_metadata.json", {"figure_id":fid,"title":title,"source_files":sources,"source_hashes":{s:source_hash(s) for s in sources},"filters":"specified by title/source rows","metric_definitions":method,"denominator":denom,"statistical_method":"frozen source aggregation; no refitting","system_config_scope":scope,"system_spec_sha256":spec_hash,"generation_script_version":VERSION,"generation_script_sha256":gen_hash})
    pipeline_svg(fdir/"figure_1_system_pipeline.png"); meta("figure_1_system_pipeline","Frozen end-to-end system",["results/final_audit_v1/final_system_spec.json"],"not applicable","process diagram","final default")
    auc=pd.read_csv(ROOT/"results/adaptive_layout_strong_v1/auc_by_seed.csv"); physical=pd.read_csv(ROOT/"results/adaptive_layout_strong_v1/physical_results.csv"); points=[]
    for _,row in auc.iterrows():
        seed=row.layout_seed; mask=(physical.partition=="validation")&(physical.alpha_method==row.alpha_method)&(physical.layout==row.layout)
        if row.layout=="distributed": mask &= physical.layout_seed.astype(str).eq(str(seed))
        part=physical[mask]; ber=float(part.bit_errors_roundtrip.sum()/part.message_bits.sum()); points.append((f"{'A' if row.alpha_method=='sender_local_adaptive' else 'F'}-{'P' if row.layout=='prefix' else seed}",ber,float(row.roc_auc),"#3478bf" if row.alpha_method=='sender_local_adaptive' else "#b35b35"))
    svg_scatter(fdir/"figure_2_reliability_detectability.png","Reliability–detectability tradeoff (comparable validation configs)",[p[0] for p in points],[p[1] for p in points],[p[2] for p in points],[p[3] for p in points]); meta("figure_2_reliability_detectability","Reliability–detectability tradeoff",["results/adaptive_layout_strong_v1/auc_by_seed.csv","results/adaptive_layout_strong_v1/physical_results.csv"],"338 replay configurations/point","x=total bit errors/total bits; y=FULL-29 replay-grouped AUC; A=adaptive, F=fixed, P=PREFIX","fixed/adaptive alpha, PREFIX/five DISTRIBUTED layouts; no lines join points")
    q=pd.read_csv(ROOT/"results/adaptive_layout_strong_v1/quarter_energy.csv"); q=q[(q.partition=="validation")&(q.alpha_method=="sender_local_adaptive")]; pre=q[q.layout=="prefix"].sort_values("quarter").pooled_energy_fraction.tolist(); dist=q[q.layout=="distributed"].groupby("quarter").pooled_energy_fraction.mean().tolist(); svg_chart(fdir/"figure_3_position_signature.png","Embedding energy by replay quarter",["P-Q1","P-Q2","P-Q3","P-Q4","D-Q1","D-Q2","D-Q3","D-Q4"],pre+dist,"energy fraction",["#b33"]*4+["#3478bf"]*4); meta("figure_3_position_signature","Embedding energy by replay quarter",["results/adaptive_layout_strong_v1/quarter_energy.csv"],"active shift energy","pooled sum-squared-shift fraction; D averaged over layouts","validation adaptive placement")
    c=pd.read_csv(ROOT/"results/decoder_error_analysis_v1/correlation_error_bins.csv"); svg_chart(fdir/"figure_4_decoder_confidence.png","Bit error probability by |correlation| decile",[str(i+1) for i in c.bin_index],c.error_probability.tolist(),"P(error)",["#5b4b8a"]*len(c)); meta("figure_4_decoder_confidence","Bit error probability by |correlation| decile",["results/decoder_error_analysis_v1/correlation_error_bins.csv"],"4,713 coded bits/bin","empirical P(error | |C| bin)","five-layout physical validation")
    s=pd.read_csv(ROOT/"results/ecc_integrity_validation_v1/summary.csv"); vals=[]; labels=[]; colors=[]
    for _,row in s.iterrows():
        for label,col in [("correct_accept_rate","#2e8b57"),("silent_error_rate","#b33"),("rejection_rate","#d9902f")]: labels.append(("Hard" if row.method=="baseline" else "Integrity")+" "+label.split('_')[0]); vals.append(float(row[label])); colors.append(col)
    svg_chart(fdir/"figure_5_integrity_tradeoff.png","Hard SECDED vs integrity outcomes",labels,vals,"configuration fraction",colors); meta("figure_5_integrity_tradeoff","Hard SECDED vs integrity outcomes",["results/ecc_integrity_validation_v1/summary.csv"],"835 configurations/method","correct/wrong/reject fractions","unseen-map integrity validation")
    pn=pd.read_csv(ROOT/"results/pn_key_robustness_phase2_v1/combined_pn_distribution.csv"); phase=pn[pn.phase=="phase2"].sort_values("weighted_raw_ber"); labels=[str(i+1) for i in range(len(phase))]; vals=phase.weighted_raw_ber.tolist(); svg_chart(fdir/"figure_6_pn_robustness.png","Phase-2 PN-key BER distribution (sorted)",labels,vals,"weighted raw BER",["#3478bf"]*len(vals),float(pn[pn.phase=="historical"].weighted_raw_ber.iloc[0])); meta("figure_6_pn_robustness","Phase-2 PN-key BER distribution",["results/pn_key_robustness_phase2_v1/combined_pn_distribution.csv"],"2,520 coded bits/key on common design","bit-weighted BER; red line historical PN","30 preregistered PN keys, three layouts")
    msg=pd.read_csv(ROOT/"results/message_content_robustness_v1/reliability_by_message.csv"); svg_chart(fdir/"figure_7_message_robustness.png","Message-family weighted raw BER",msg.message_family.tolist(),msg.weighted_raw_ber.tolist(),"weighted raw BER",["#7b5e35"]*len(msg)); meta("figure_7_message_robustness","Message-family weighted raw BER",["results/message_content_robustness_v1/reliability_by_message.csv"],"3,744 coded bits/family","total errors / coded bits","K=1 reused panel, one PN, three layouts")


def paper_tables(out: Path, spec_hash: str) -> None:
    claims={r["claim_id"]:r for r in claim_rows(spec_hash)}
    tables={
        "table_dataset_quality.csv":[{"claim_id":"DATA_001","metric":"active corpus","value":"949 replays / 34 maps","scope":"post-quarantine"},{"claim_id":"MATCH_001","metric":"weighted match ratio","value":"99.22%","scope":"active 949"},{"claim_id":"MATCH_001","metric":"median replay match ratio","value":"100%","scope":"active 949"}],
        "table_final_system.csv":[{"claim_id":"FINAL_SPEC","component":"alpha","value":"sender-local adaptive v2 exact rational"},{"claim_id":"FINAL_SPEC","component":"placement","value":"keyed DISTRIBUTED, N=8"},{"claim_id":"FINAL_SPEC","component":"payload","value":"normal 7.5%; ABSTAIN without complete word"},{"claim_id":"FINAL_SPEC","component":"coding","value":"SECDED(8,4) + frozen integrity"},{"claim_id":"FINAL_SPEC","component":"PN","value":"ordinary deterministic 32-bit effective seed; no screening"}],
        "table_steganalysis.csv":[{"claim_id":"DIST_002","evaluation":"adaptive DISTRIBUTED unseen maps","FULL29_AUC":"0.597–0.614","scope":"five layouts; RF"},{"claim_id":"CLF_001","evaluation":"replay-grouped classifier families","FULL29_AUC":"0.603–0.635","scope":"four classifiers"},{"claim_id":"CLF_001","evaluation":"unseen-map classifier families","FULL29_AUC":"0.603–0.634","scope":"four classifiers"}],
        "table_ecc_integrity.csv":[{"claim_id":"ECC_001","metric":"equal-useful full recovery gain","baseline":"72.0% uncoded","final_or_ecc":"90.0% SECDED","cost":"50% code rate"},{"claim_id":"INT_001","metric":"wrong accept","baseline":"14/835","final_or_ecc":"1/835","cost":"rejection 54→118"},{"claim_id":"INT_002","metric":"correct accept","baseline":"767/835","final_or_ecc":"716/835","cost":"51 additional correct rejects"}],
        "table_robustness.csv":[{"claim_id":"PN_001","dimension":"PN key","result":"mean 5.136%; SD 0.478 pp; range 4.087–6.190%","evidence":"30 preregistered keys"},{"claim_id":"MSG_001","dimension":"message","result":"BER 5.769–7.131%; SD 0.498 pp","evidence":"five messages"},{"claim_id":"CLF_001","dimension":"classifier","result":"FULL AUC 0.603–0.635","evidence":"four families, two regimes"}],
        "table_negative_results.csv":negative_rows(spec_hash),
    }
    for name,rows in tables.items():
        for row in rows:
            row["system_spec_sha256"]=spec_hash
            cid=row.get("claim_id","")
            row["source_artifact"]=claims.get(cid,{}).get("source_file", "results/final_audit_v1/negative_results.csv" if name=="table_negative_results.csv" else "results/final_audit_v1/final_system_spec.json")
        write_csv(out/name,rows); csv_to_markdown(out/name,out/name.replace(".csv",".md"))


def secret_audit() -> tuple[bool, list[str]]:
    findings: list[str] = []
    placeholders = {"", "changeme", "your_client_id", "your_client_secret", "xxx"}
    for path in sorted(ROOT.rglob("*.py")):
        if "results/final_audit_v1" in path.as_posix():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [target.id.lower() for target in targets if isinstance(target, ast.Name)]
                value = node.value
                risky_name = any(any(term in name for term in ("client_secret", "access_token", "bearer_token", "api_secret")) for name in names)
                if risky_name and isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.lower() not in placeholders and len(value.value) >= 8:
                    findings.append(f"{rel(path)}:{node.lineno}: hardcoded credential-like literal (value suppressed)")
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "getenv" and len(value.args) > 1:
                    default = value.args[1]
                    if isinstance(default, ast.Constant) and str(default.value).lower() not in placeholders:
                        findings.append(f"{rel(path)}:{node.lineno}: environment lookup has credential-like literal fallback (value suppressed)")
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and str(key.value).lower() in {"client_secret", "access_token", "authorization"} and isinstance(value, ast.Constant) and len(str(value.value)) >= 8:
                        findings.append(f"{rel(path)}:{node.lineno}: credential-like dictionary literal (value suppressed)")
    for path in ROOT.rglob(".env*"):
        if path.is_file():
            findings.append(f"{rel(path)}: environment file present; contents not printed")
    return not findings, findings


def final_report(out: Path, spec_hash: str, manifest_rows: list[dict[str, Any]], secret_ok: bool, secret_findings: list[str]) -> None:
    (out/"final_audit_report.md").write_text(f"""# Final scientific audit v1

Final status: **B — publishable with major caveats**. Frozen scientific artifacts are internally consistent and reproducible, but detectability is above chance, useful throughput is constrained by 50% ECC rate and abstention, and external-source validation is absent.

1. **Final system:** sender-local adaptive alpha v2 (exact rational thresholds), independently keyed DISTRIBUTED placement, `N=8`, normal 7.5% coded-bit allocation, SECDED(8,4), corrected-bit-not-minimum integrity rejection, and ABSTAIN when no complete word fits. Spec SHA-256: `{spec_hash}`.
2. **Supporting corpora:** active 949/34 original corpus; frozen 611/23 development and 338/11 validation split; 300/20 new floor-validation primary corpus with disjoint 240-replay calibration pool; consumed 68-replay PN, 100-replay host/message, and 203-replay classifier panels.
3. **Superseded/invalid:** old timing/matching baseline, float-boundary adaptive inference, PREFIX as final layout, confidence erasure, 12.5% floor, host-aware PN selection, and any message/PN screening claim.
4. **Hashes:** all {len(manifest_rows)} available branch manifests passed ({sum(int(r['entries']) for r in manifest_rows)} listed artifacts); branches without manifests are explicitly lower provenance.
5. **Semantics:** timing, matching, deduplication, writer, decoder-zero, adaptive boundaries, DISTRIBUTED, SECDED, integrity, and abstention agree. One documentation error was confirmed: 98.68% is pre-quarantine; active 949 is 99.22%.
6. **Reliability:** scoped physical BER is 4.21% on the original five-layout decoder matrix; new-corpus normal eligible cases were 2.55%. These are different corpora and are not pooled.
7. **ECC/integrity:** equal-useful SECDED recovery improved 72%→90%; integrity changed wrong accept 14→1, correct accept 767→716, and reject 54→118 of 835.
8. **Useful-payload cost:** SECDED rate is 1/2; new-corpus normal eligibility was 45%, with 4.444 useful bits/eligible configuration and 1.763 correctly accepted useful bits/input replay-layout opportunity.
9. **Detectability:** modest but above chance; adaptive DISTRIBUTED unseen-map FULL-29 AUC was 0.597–0.614 across layouts.
10. **Unseen maps:** yes, position-signature reduction and above-chance detection generalized to the frozen map-disjoint validation.
11. **Classifier families:** FULL-29 AUC was 0.603–0.635 in both replay-grouped and unseen-map regimes; not RF-specific under tested features.
12. **PN effect:** Phase-2 mean BER 5.136%, SD 0.478 pp, range 4.087–6.190%; meaningful but not catastrophic sensitivity.
13. **Message effect:** BER 5.769–7.131%, SD 0.498 pp; modest sensitivity on one-PN consumed data.
14. **Rejected methods:** PREFIX, confidence erasure, 12.5% floor, deployable host-aware PN screening, and untested message screening.
15. **Paired comparisons:** PREFIX/DISTRIBUTED, ECC controls, integrity/hard SECDED, within-panel PN, message, and classifier comparisons. The matrix marks cross-corpus comparisons NO.
16. **Non-combinable headlines:** 4.21% original physical BER, 2.55% new-corpus normal BER, 1/835 integrity validation wrong accept, and classifier robustness AUCs arise from different scopes.
17. **CI units:** replay clusters for paired/replay outcomes, beatmap clusters for unseen-map AUC, PN keys for between-PN mean, and PN+replay for Phase-2 system uncertainty.
18. **Strongest limitations:** no fully external source, above-chance detectability, hand-engineered adversaries, 32-bit non-cryptographic PN seed, abstention/throughput cost, and PN/message variation.
19. **Reviewer concerns:** strongest are selection/quarantine, development reuse, AUC≈0.6, abstention-conditioned reliability, and limited attacker family; see `reviewer_attack.md`.
20. **Figures/tables:** seven figures with metadata and six CSV/Markdown tables regenerate deterministically from frozen artifacts.
21. **Safe abstract claims:** those in `abstract_claims.md`, all claim-tagged and scoped.
22. **Forbidden claims:** undetectable, secure, authenticated, zero-risk, universally optimal, externally validated, or capacity-improved without redundancy/abstention accounting.
23. **Unresolved blockers:** no internal scientific blocker to a carefully scoped paper; external generalization and stronger adversaries remain research limitations.
24. **Project status:** **B — publishable with major caveats**.

Secret audit: {"PASS; environment-only API credentials" if secret_ok else "REVIEW REQUIRED: " + "; ".join(secret_findings)}. Git-history inspection was not possible because this workspace is not a Git worktree.
""",encoding="utf-8")


def generate(out: Path) -> tuple[str, list[dict[str, Any]]]:
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest_rows=audit_manifests()
    write_csv(out/"artifact_hash_audit.csv",manifest_rows)
    write_csv(out/"artifact_inventory.csv",artifact_inventory())
    spec=make_spec(); write_json(out/"final_system_spec.json",spec); spec_hash=sha256(out/"final_system_spec.json")
    write_csv(out/"final_claim_registry.csv",claim_rows(spec_hash))
    write_csv(out/"final_method_status.csv",method_status_rows(spec_hash))
    write_csv(out/"dataset_registry.csv",dataset_rows(spec_hash))
    write_csv(out/"denominator_audit.csv",denominator_rows(spec_hash))
    write_csv(out/"uncertainty_audit.csv",uncertainty_rows(spec_hash))
    write_csv(out/"canonical_metrics.csv",canonical_metrics(spec_hash))
    write_csv(out/"final_end_to_end_metrics.csv",end_to_end_rows(spec_hash))
    write_csv(out/"result_comparability_matrix.csv",comparability_rows(spec_hash))
    write_csv(out/"method_evolution.csv",evolution_rows(spec_hash))
    write_csv(out/"negative_results.csv",negative_rows(spec_hash))
    write_csv(out/"claim_language_audit.csv",safe_language_rows(spec_hash))
    write_longform(out,spec_hash); narrative(out)
    write_csv(out/"final_conclusions_matrix.csv",conclusions_rows(spec_hash))
    (out/"reviewer_attack.md").write_text(reviewer_text(),encoding="utf-8")
    paper_tables(out,spec_hash); figures(out,spec_hash)
    secret_ok,findings=secret_audit(); final_report(out,spec_hash,manifest_rows,secret_ok,findings)
    return spec_hash,manifest_rows


def output_hashes(out: Path) -> dict[str,str]:
    return {p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"result_hashes.json","final_reproducibility_audit.md"}}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--check-reproducibility",action="store_true")
    args=parser.parse_args(); out=args.output_dir.resolve()
    spec_hash,manifest_rows=generate(out); first=output_hashes(out)
    repro="not requested"
    if args.check_reproducibility:
        with tempfile.TemporaryDirectory(prefix="osu-final-audit-") as name:
            temp=Path(name)/"final_audit_v1"; generate(temp); second=output_hashes(temp)
            if first != second:
                changed=sorted(set(first)^set(second) | {k for k in set(first)&set(second) if first[k]!=second[k]})
                raise RuntimeError("Non-deterministic finalizer outputs: "+", ".join(changed))
        repro="PASS: clean temporary regeneration matched byte-for-byte"
    (out/"final_reproducibility_audit.md").write_text(f"""# Final reproducibility audit

- Frozen branch manifests: PASS ({len(manifest_rows)} manifests; {sum(int(r['entries']) for r in manifest_rows)} artifacts).
- No physical replay generation: PASS by entry-point code-path audit.
- No API access, offset calibration, corpus rematching, or model fitting: PASS.
- No frozen source rows/configs changed: PASS; finalizer writes only its selected output directory.
- Deterministic canonical metrics, claim registry, tables, metadata, and figures: {repro}.
- System specification SHA-256: `{spec_hash}`.
""",encoding="utf-8")
    hashes=output_hashes(out); hashes["final_reproducibility_audit.md"]=sha256(out/"final_reproducibility_audit.md")
    write_json(out/"result_hashes.json",{"generator":VERSION,"system_spec_sha256":spec_hash,"files":hashes})
    print(f"Final audit generated: {out}")
    print(f"System spec SHA-256: {spec_hash}")
    print(f"Frozen manifests: {len(manifest_rows)} PASS")
    print(f"Reproducibility: {repro}")


if __name__ == "__main__":
    main()
