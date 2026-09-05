"""Record resolved estimator defaults and hashes for finalized result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import sklearn

from osu_stego.analysis.classifier_families import CLASSIFIER_NAMES, build_classifier
from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"
SEED_RULE = (
    "stable_seed(42,classifier-family-robustness-v1,scope,classifier,"
    "feature_set,fold_or_fit)"
)


def json_value(value):
    """Convert sklearn parameter values to stable JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return repr(value)


def resolved_parameters() -> dict:
    """Resolve constructor defaults under the recorded sklearn release."""
    classifiers = {}
    for name in CLASSIFIER_NAMES:
        model = build_classifier(name, 0)
        parameters = {}
        for key, value in model.get_params(deep=True).items():
            if key in ("steps", "scale", "classifier"):
                continue
            parameters[key] = SEED_RULE if key.endswith("random_state") else json_value(value)
        classifiers[name] = {
            "estimator": type(model).__name__,
            "parameters": parameters,
        }
    return {
        "sklearn_version": sklearn.__version__,
        "random_state_rule": SEED_RULE,
        "classifiers": classifiers,
    }


def main() -> None:
    resolved_path = OUTPUT / "resolved_classifier_parameters.json"
    resolved_path.write_text(
        json.dumps(resolved_parameters(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: file_sha256(path)
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "result_hashes.json"
    }
    (OUTPUT / "result_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"finalized {len(hashes)} classifier-family result artifacts")


if __name__ == "__main__":
    main()
