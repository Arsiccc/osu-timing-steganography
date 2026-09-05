"""Hash finalized message-content robustness artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "message_content_robustness_v1"


def main() -> None:
    hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "result_hashes.json"
    }
    payload = json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    target = OUTPUT / "result_hashes.json"
    changed = not target.exists() or target.read_text(encoding="utf-8") != payload
    target.write_text(payload, encoding="utf-8")
    print(f"finalized {len(hashes)} artifacts; changed={int(changed)}")


if __name__ == "__main__":
    main()
