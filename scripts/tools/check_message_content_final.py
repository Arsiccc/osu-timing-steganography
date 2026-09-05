"""Final post-correction checks for message-content robustness v1."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "message_content_robustness_v1"


def main() -> None:
    correction = json.loads((OUTPUT / "post_lock_corrections.json").read_text())
    symmetry = pd.read_csv(OUTPUT / "complement_symmetry.csv")
    if not np.isclose(symmetry.coded_hamming_fraction_weighted.iloc[0], 1.0):
        raise AssertionError("All-zero/all-one are not recorded as exact complements.")
    if not (
        symmetry.raw_ber_difference_ci95_low.iloc[0]
        <= symmetry.raw_ber_difference.iloc[0]
        <= symmetry.raw_ber_difference_ci95_high.iloc[0]
    ):
        raise AssertionError("Complement estimate lies outside its interval.")
    report = (OUTPUT / "message_content_report.md").read_text()
    if "therefore are exact coded-bit complements" not in report:
        raise AssertionError("Correct complement interpretation missing.")
    if "so they are not exact complements" in report:
        raise AssertionError("Stale complement interpretation remains.")
    if correction["physical_data_changed"] or correction["model_parameters_changed"]:
        raise AssertionError("Correction unexpectedly changes frozen methodology.")
    manifest_path = OUTPUT / "result_hashes.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, metadata in manifest.items():
            if file_sha256(OUTPUT / name) != metadata["sha256"]:
                raise AssertionError(f"Result hash mismatch: {name}")
    print("message-content final post-correction audit: PASS")


if __name__ == "__main__":
    main()
