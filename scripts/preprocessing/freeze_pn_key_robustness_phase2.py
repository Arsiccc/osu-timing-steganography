"""Create the immutable Phase-2 experiment lock before physical outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT=RESULTS_DIR/"pn_key_robustness_phase2_v1"


def main() -> None:
    target=OUTPUT/"experiment_lock.json"
    if target.exists(): raise FileExistsError("Phase-2 experiment lock already exists.")
    config_path=CONFIG_DIR/"pn_key_robustness_phase2_v1.json"; config=json.loads(config_path.read_text()); panel=pd.read_csv(OUTPUT/"replay_panel.csv")
    paths={"runner":Path("scripts/experiments/run_pn_key_robustness_phase2.py"),"analysis":Path("scripts/analysis/analyze_pn_key_robustness_phase2.py"),
        "preparation":Path("scripts/preprocessing/prepare_pn_key_robustness_phase2.py"),"sanity":Path("scripts/tools/check_pn_key_robustness_phase2.py"),
        "pn_generator":Path("osu_stego/stego/pn_sequence.py"),"payload_layout":Path("osu_stego/stego/payload_layout.py"),"ecc":Path("osu_stego/stego/ecc.py"),
        "integrity":Path("osu_stego/stego/integrity.py"),"adaptive_alpha":Path("osu_stego/stego/adaptive_alpha.py"),"writer":Path("osu_stego/parsing/osr_writer.py"),
        "matcher":Path("osu_stego/matching/matcher.py"),"decoder":Path("osu_stego/stego/decoder.py")}
    lock={"experiment_version":"pn-key-robustness-phase2-v1","lock_created_before_any_phase2_physical_outcome":True,
        "pn_config_sha256":file_sha256(config_path),"replay_panel_sha256":file_sha256(OUTPUT/"replay_panel.csv"),
        "detector_subset_lock_sha256":file_sha256(OUTPUT/"detector_subset_lock.json"),"phase0_design_audit_sha256":file_sha256(OUTPUT/"phase0_design_audit.csv"),
        "historical_key_id":config["historical_key_id"],"phase1_key_ids":config["phase1_key_ids"],"phase2_key_ids":[x["key_id"] for x in config["new_keys"]],
        "effective_seed_collision_count":0,"incoming_replays":len(panel),"eligible_replays":int(panel.eligible_7_5.sum()),"eligible_maps":int(panel.loc[panel.eligible_7_5==1,"beatmap_hash"].nunique()),
        "layout_seeds":config["layout_seeds"],"N":config["N"],"payload_fraction":config["payload_fraction"],
        "adaptive_policy_sha256":file_sha256(CONFIG_DIR/"adaptive_alpha_sender_local_v2.json"),"integrity_rule_sha256":file_sha256(CONFIG_DIR/"ecc_integrity_rejection_v1.json"),
        "map_offsets_sha256":file_sha256(CONFIG_DIR/"map_time_offsets.json"),"source_sha256":{name:file_sha256(path) for name,path in paths.items()},
        "git_commit":"unavailable_not_a_git_worktree","expected_physical_configs":int(panel.eligible_7_5.sum())*30*len(config["layout_seeds"])}
    target.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(f"experiment_lock_sha256={file_sha256(target)}")


if __name__=="__main__": main()
