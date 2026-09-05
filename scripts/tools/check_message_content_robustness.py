"""Pre-outcome checks for the locked K=1 message-content reanalysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "message_content_robustness_v1"
SOURCE = RESULTS_DIR / "host_aware_pn_selection_v1"


def main() -> None:
    lock = json.loads((OUTPUT / "experiment_lock.json").read_text())
    for name, expected in lock["source_artifact_sha256"].items():
        if file_sha256(SOURCE / name) != expected:
            raise AssertionError(f"Source artifact changed: {name}")
    for name, expected in lock["analysis_input_sha256"].items():
        if file_sha256(OUTPUT / name) != expected:
            raise AssertionError(f"Locked analysis input changed: {name}")
    for name, expected in lock["source_code_sha256"].items():
        paths = {
            "preparation": "scripts/preprocessing/prepare_message_content_robustness.py",
            "freeze": "scripts/preprocessing/freeze_message_content_robustness.py",
            "analysis": "scripts/analysis/analyze_message_content_robustness.py",
            "sanity": "scripts/tools/check_message_content_robustness.py",
            "result_audit": "scripts/tools/check_message_content_results.py",
            "finalizer": "scripts/analysis/finalize_message_content_robustness.py",
            "message_generator": "osu_stego/stego/host_aware_pn.py",
            "ecc": "osu_stego/stego/ecc.py", "integrity": "osu_stego/stego/integrity.py",
            "adaptive_alpha": "osu_stego/stego/adaptive_alpha.py",
            "payload_layout": "osu_stego/stego/payload_layout.py",
            "pn_generator": "osu_stego/stego/pn_sequence.py",
            "writer": "osu_stego/parsing/osr_writer.py",
            "matcher": "osu_stego/matching/matcher.py",
            "timing_features": "osu_stego/analysis/timing_features.py",
            "classifier_families": "osu_stego/analysis/classifier_families.py",
        }
        if file_sha256(Path(paths[name])) != expected:
            raise AssertionError(f"Locked source changed: {name}")

    known = pd.read_csv(SOURCE / "index_known_results.csv")
    k1 = known[known.K == 1].copy()
    if len(k1) != 1500 or k1.replay_file.nunique() != 100:
        raise AssertionError("Unexpected K=1 condition population.")
    if set(k1.message_family) != set(lock["message_families"]):
        raise AssertionError("Message families changed.")
    if set(k1.layout_seed.astype(int)) != {0, 1, 2}:
        raise AssertionError("Layout population changed.")
    if set(k1.selected_pn_index.astype(int)) != {0}:
        raise AssertionError("K=1 does not fix candidate zero.")
    if set(k1.selected_pn_id.astype(str)) != {lock["fixed_physical_configuration"]["pn_key_id"]}:
        raise AssertionError("K=1 PN identity changed.")
    if k1.duplicated(["replay_file", "message_family", "layout_seed"]).any():
        raise AssertionError("Duplicate K=1 condition identity.")
    if not np.all(k1.groupby("replay_file").size().to_numpy() == 15):
        raise AssertionError("Incomplete within-replay repeated measures.")
    if k1.groupby("replay_file").alpha.nunique().max() != 1:
        raise AssertionError("Alpha differs within replay.")
    if k1.groupby("replay_file")[["coded_bits", "useful_bits"]].nunique().max().max() != 1:
        raise AssertionError("Payload length differs within replay.")
    bits = pd.read_csv(SOURCE / "bit_results.csv").merge(
        k1[["physical_id"]], on="physical_id", validate="many_to_one"
    )
    if bits.groupby(["replay_file", "layout_seed", "bit_index"]).block_index.nunique().max() != 1:
        raise AssertionError("DISTRIBUTED block identity differs by message.")
    if not np.isfinite(bits[["correlation", "abs_correlation"]].to_numpy(float)).all():
        raise AssertionError("Non-finite K=1 decoder scores.")
    folds = pd.read_csv(OUTPUT / "fold_manifest.csv")
    if folds.replay_file.nunique() != 100 or folds.fold.nunique() != 5:
        raise AssertionError("Invalid common replay folds.")
    print("message-content pre-outcome checks: PASS")


if __name__ == "__main__":
    main()
