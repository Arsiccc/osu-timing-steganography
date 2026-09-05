"""Pre-outcome Phase-2 design audit, PN preregistration, and panel freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from osu_stego.stego.pn_sequence import _derive_seed
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, N_VALUE, PAYLOAD_FRACTION, PN_KEY
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import key_id


VERSION = "pn-key-robustness-phase2-v1"
MASTER = "osu-stego|pn-key-robustness-phase2-v1|preregistered-2026-09-02"
OUTPUT = RESULTS_DIR / "pn_key_robustness_phase2_v1"
PHASE1 = RESULTS_DIR / "pn_key_robustness_v1"
CONFIG = CONFIG_DIR / "pn_key_robustness_phase2_v1.json"
LAYOUT_INDICES = (0, 1, 2)
DETECTOR_INDICES = (0, 5, 10, 15, 20, 25, 29)


def derive_key(index: int) -> str:
    label = f"pn_robustness_phase2_{index:02d}"
    digest = hashlib.sha256(f"{MASTER}|{label}".encode()).hexdigest()
    return f"{VERSION}:{label}:{digest}"


def phase0_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = pd.read_csv(PHASE1 / "physical_results.csv")
    full = physical.groupby("pn_key_id").agg(errors=("raw_bit_errors", "sum"), bits=("coded_bits", "sum"))
    full["ber"] = full.errors / full.bits
    rows = []
    for count in range(1, 6):
        subset = physical[physical.layout_seed < count].groupby("pn_key_id").agg(errors=("raw_bit_errors", "sum"), bits=("coded_bits", "sum"))
        subset["ber"] = subset.errors / subset.bits
        x, y = subset.ber.reindex(full.index), full.ber
        rows.append({"first_k_layouts": count, "pearson_with_five": x.corr(y), "spearman_with_five": x.corr(y, method="spearman"),
                     "mean_absolute_error_pp": float(np.mean(np.abs(x-y))*100), "max_absolute_error_pp": float(np.max(np.abs(x-y))*100),
                     "between_key_sd_pp": float(x.std(ddof=1)*100)})
    matrix = physical.groupby(["pn_key_id", "layout_seed"]).agg(errors=("raw_bit_errors", "sum"), bits=("coded_bits", "sum"))
    matrix["weighted_ber"] = matrix.errors / matrix.bits
    return pd.DataFrame(rows), matrix.reset_index()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = [CONFIG, OUTPUT/"phase2_pn_keys.json", OUTPUT/"replay_panel.csv", OUTPUT/"phase0_design_audit.csv", OUTPUT/"detector_subset_lock.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Phase-2 preregistration artifact već postoji; odbijam overwrite.")
    audit, matrix = phase0_audit()
    audit.to_csv(OUTPUT/"phase0_design_audit.csv", index=False)
    matrix.to_csv(OUTPUT/"phase0_pn_layout_matrix.csv", index=False)
    (OUTPUT/"phase0_design_recommendation.md").write_text(
        "# Phase-0 design recommendation\n\nUse the unchanged Phase-1 panel and the first three frozen layout seeds (0, 1, 2). "
        "One layout was too noisy; two retained useful rank correlation but had materially larger error versus the five-layout estimate; three provided the safer compromise. "
        "The choice is deterministic and was frozen before generating or evaluating Phase-2 keys.\n", encoding="utf-8")

    panel = pd.read_csv(PHASE1/"pilot_selection.csv")
    panel.to_csv(OUTPUT/"replay_panel.csv", index=False)
    phase1 = json.loads((PHASE1/"pn_keys.json").read_text())
    new_keys = [derive_key(i) for i in range(30)]
    prior = [phase1["historical_reference"], *phase1["new_keys"]]
    records = [{"index": i, "label": f"pn_robustness_phase2_{i:02d}", "key": key, "key_id": key_id(key), "effective_seed_32": _derive_seed(key)} for i,key in enumerate(new_keys)]
    all_seed_rows = [{"phase": "historical_or_phase1", "key_id": row["key_id"], "effective_seed_32": _derive_seed(row["key"])} for row in prior]
    all_seed_rows += [{"phase": "phase2", "key_id": row["key_id"], "effective_seed_32": row["effective_seed_32"]} for row in records]
    collisions = pd.DataFrame(all_seed_rows).groupby("effective_seed_32").filter(lambda x: len(x)>1)
    collisions.to_csv(OUTPUT/"effective_seed_collisions.csv", index=False)
    if not collisions.empty:
        raise RuntimeError("Preregistered effective PN seed collision; artifacts retained, STOP required.")
    config = {"version": VERSION, "scientific_status": "preregistered_before_phase2_outcomes", "master_derivation_rule": "key=version:label:SHA256(master|label)",
              "master_material": MASTER, "master_sha256": hashlib.sha256(MASTER.encode()).hexdigest(),
              "historical_key_id": phase1["historical_reference"]["key_id"], "phase1_key_ids": [x["key_id"] for x in phase1["new_keys"]],
              "new_keys": records, "pn_implementation_sha256": file_sha256(Path("osu_stego/stego/pn_sequence.py")),
              "N": N_VALUE, "payload_fraction": PAYLOAD_FRACTION, "layout_seeds": [{"seed":i,"key_id":key_id(LAYOUT_KEYS[i])} for i in LAYOUT_INDICES],
              "replay_panel_sha256": file_sha256(OUTPUT/"replay_panel.csv"), "incoming_replays": len(panel), "eligible_replays": int(panel.eligible_7_5.sum()),
              "adaptive_policy_sha256": file_sha256(CONFIG_DIR/"adaptive_alpha_sender_local_v2.json"),
              "integrity_rule_sha256": file_sha256(CONFIG_DIR/"ecc_integrity_rejection_v1.json")}
    CONFIG.write_text(json.dumps(config,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (OUTPUT/"phase2_pn_keys.json").write_bytes(CONFIG.read_bytes())
    detector = {"status":"preregistered_before_phase2_ber_outcomes", "phase2_key_indices":list(DETECTOR_INDICES),
                "phase2_key_ids":[records[i]["key_id"] for i in DETECTOR_INDICES], "include_historical_reference":True,
                "feature_sets":["baseline","position_only","full"], "rf":"frozen Phase-1 configuration", "layout_seeds":list(LAYOUT_INDICES)}
    (OUTPUT/"detector_subset_lock.json").write_text(json.dumps(detector,indent=2,sort_keys=True)+"\n")
    print(f"design=68 eligible x 30 keys x 3 layouts = {68*30*3}")
    print(f"pn_config_sha256={file_sha256(CONFIG)}")
    print(f"panel_sha256={file_sha256(OUTPUT/'replay_panel.csv')}")


if __name__=="__main__": main()
