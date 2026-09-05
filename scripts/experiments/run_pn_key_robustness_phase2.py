"""Locked Phase-2 physical run across 30 preregistered PN keys."""

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
from scripts.experiments.run_pn_key_robustness import append_unique


VERSION = "pn-key-robustness-phase2-v1"
OUTPUT = RESULTS_DIR / "pn_key_robustness_phase2_v1"


def identity(replay: str, replay_sha: str, pn_id: str, seed: int, encoded: np.ndarray) -> str:
    material = "|".join((VERSION, replay, replay_sha, pn_id, str(seed), array_hash(encoded)))
    return hashlib.sha256(material.encode()).hexdigest()


def completed(output: Path) -> set[str]:
    found=[]
    for name,count in (("physical_results.csv",1),("message_results.csv",1),("timing_features.csv",2)):
        path=output/name
        if not path.exists(): found.append(set()); continue
        counts=pd.read_csv(path).groupby("config_id").size(); found.append(set(counts[counts==count].index.astype(str)))
    return set.intersection(*found)


def flush(output: Path, buffers: dict[str,list[dict]]) -> None:
    specs={"physical":("physical_results.csv",("config_id",)),"message":("message_results.csv",("config_id",)),
           "codeword":("codeword_results.csv",("config_id","codeword_index")),"bit":("bit_results.csv",("config_id","bit_index")),
           "features":("timing_features.csv",("config_id","label")),"wrong":("wrong_key_sanity.csv",("config_id",))}
    for name,(filename,keys) in specs.items(): append_unique(output/filename,buffers[name],keys); buffers[name].clear()


def load_locked(args: argparse.Namespace) -> tuple[dict,dict,pd.DataFrame]:
    lock=json.loads(args.lock.read_text()); config=json.loads(args.pn_config.read_text())
    checks=((args.pn_config,"pn_config_sha256"),(args.panel,"replay_panel_sha256"),(args.adaptive_policy,"adaptive_policy_sha256"),
            (args.integrity_rule,"integrity_rule_sha256"),(args.map_offsets,"map_offsets_sha256"))
    for path,field in checks:
        if file_sha256(path)!=lock[field]: raise RuntimeError(f"Locked hash mismatch: {path}")
    if file_sha256(Path(__file__))!=lock["source_sha256"]["runner"]: raise RuntimeError("Runner changed after lock.")
    return config,load_sender_local_policy(args.adaptive_policy),pd.read_csv(args.panel)


def run(args: argparse.Namespace, config: dict, adaptive: dict, panel: pd.DataFrame) -> int:
    selected=panel[panel.eligible_7_5==1].sort_values("selection_order")
    if args.limit_replays: selected=selected.head(args.limit_replays)
    keys=config["new_keys"]; layout_indices=[int(x["seed"]) for x in config["layout_seeds"]]
    done=completed(args.output_dir); beatmaps=build_beatmap_index(args.dataset_dir); offsets=load_map_offsets(args.map_offsets)
    buffers={name:[] for name in ("physical","message","codeword","bit","features","wrong")}; new=0
    with tempfile.TemporaryDirectory(prefix="pn_phase2_") as temp:
        for replay_number,source in enumerate(selected.itertuples(index=False),1):
            context=prepare_replay(source,args.dataset_dir,beatmaps,offsets); alpha=alpha_from_replay(adaptive,context.replay)
            useful=deterministic_message(context.replay_file,context.beatmap_hash,0,N_VALUE,PAYLOAD_FRACTION,int(source.useful_bits),MESSAGE_SEED)
            encoded=encode_hamming_8_4_secded(useful); replay_sha=file_sha256(context.osr_path); clean=residual_timing_features(context.original_residuals)
            for pn_index,pn in enumerate(keys):
                wrong=keys[(pn_index+1)%len(keys)]
                for seed in layout_indices:
                    layout_key=LAYOUT_KEYS[seed]; cid=identity(context.replay_file,replay_sha,pn["key_id"],seed,encoded)
                    if cid in done: continue
                    _,roundtrip,diag=build_physical_stego(context,encoded,alpha,N_VALUE,"distributed",pn["key"],layout_key,
                        float(adaptive["hit_margin_ms"]),Path(temp)/f"{cid}.osr")
                    received=extract_bipolar_message_with_layout(roundtrip,pn["key"],layout_key,N_VALUE,len(encoded),"distributed")
                    corr=message_correlations_with_layout(roundtrip,pn["key"],layout_key,N_VALUE,len(encoded),"distributed")
                    decoded,statuses=decode_hamming_8_4_secded(received); decisions=[]; word_correct=[]
                    common={"experiment_version":VERSION,"config_id":cid,"replay_file":context.replay_file,"beatmap_hash":context.beatmap_hash,
                        "performance_category":source.performance_category,"pn_key_index":pn_index,"pn_key_id":pn["key_id"],"effective_seed_32":pn["effective_seed_32"],
                        "layout_seed":seed,"layout_key_id":key_id(layout_key),"alpha":alpha,"N":N_VALUE,"payload_fraction":PAYLOAD_FRACTION,
                        "coded_bits":len(encoded),"useful_bits":len(useful),"message_sha256":array_hash(useful),"encoded_sha256":array_hash(encoded),"replay_sha256":replay_sha}
                    for word in range(len(encoded)//8):
                        cs=slice(word*8,word*8+8); us=slice(word*4,word*4+4); syndrome,parity=secded_syndrome(received[cs])
                        decision=corrected_bit_is_not_minimum_decision(str(statuses[word]),syndrome,parity,np.abs(corr[cs]))
                        correct=bool(np.array_equal(decoded[us],useful[us])); decisions.append(decision); word_correct.append(correct); rank=np.nan
                        if str(statuses[word])=="corrected_single":
                            pos=syndrome-1 if syndrome else 7; rank=int(np.argsort(np.argsort(np.abs(corr[cs]),kind="stable"),kind="stable")[pos])
                        buffers["codeword"].append({**common,"codeword_index":word,"hard_status":str(statuses[word]),"raw_error_count":int(np.sum(received[cs]!=encoded[cs])),
                            "word_correct":int(correct),"integrity_accepted":int(decision.accepted),"integrity_reason":decision.reason,"corrected_bit_confidence_rank":rank})
                    accepted,correct=all(x.accepted for x in decisions),all(word_correct); outcome="REJECT" if not accepted else ("CORRECT_ACCEPT" if correct else "WRONG_ACCEPT")
                    buffers["message"].append({**common,"outcome":outcome,"correct_accept":int(outcome=="CORRECT_ACCEPT"),"wrong_accept":int(outcome=="WRONG_ACCEPT"),
                        "reject":int(outcome=="REJECT"),"post_ecc_bit_errors":int(np.sum(decoded!=useful)),"post_ecc_ber":float(np.mean(decoded!=useful))})
                    requested=int(diag["requested_carriers"]); active=int(diag["active_carriers"])
                    buffers["physical"].append({**common,"raw_bit_errors":int(np.sum(received!=encoded)),"raw_ber":float(np.mean(received!=encoded)),
                        "active_carrier_fraction":active/requested,**diag})
                    bit_rows,_=bit_level_diagnostics(context=context,roundtrip_residuals=roundtrip,message=encoded,alpha=alpha,n_frames_per_bit=N_VALUE,
                        pn_key=pn["key"],layout_key=layout_key,hit_margin_ms=float(adaptive["hit_margin_ms"]))
                    buffers["bit"].extend({**common,**row} for row in bit_rows)
                    for label,values in ((0,clean),(1,residual_timing_features(roundtrip))): buffers["features"].append({**common,"label":label,"feature_version":FEATURE_VERSION,**values})
                    wrong_bits=extract_bipolar_message_with_layout(roundtrip,wrong["key"],layout_key,N_VALUE,len(encoded),"distributed")
                    wrong_corr=message_correlations_with_layout(roundtrip,wrong["key"],layout_key,N_VALUE,len(encoded),"distributed")
                    buffers["wrong"].append({**common,"wrong_pn_key_id":wrong["key_id"],"wrong_key_bit_errors":int(np.sum(wrong_bits!=encoded)),
                        "wrong_key_ber":float(np.mean(wrong_bits!=encoded)),"wrong_key_mean_abs_correlation":float(np.mean(np.abs(wrong_corr)))})
                    new+=1
            if replay_number%5==0: flush(args.output_dir,buffers); print(f"physical {replay_number}/{len(selected)} new={new}",flush=True)
    flush(args.output_dir,buffers); return new


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir",type=Path,default=OUTPUT); parser.add_argument("--dataset-dir",type=Path,default=DATASET_DIR)
    parser.add_argument("--panel",type=Path,default=OUTPUT/"replay_panel.csv"); parser.add_argument("--pn-config",type=Path,default=CONFIG_DIR/"pn_key_robustness_phase2_v1.json")
    parser.add_argument("--lock",type=Path,default=OUTPUT/"experiment_lock.json"); parser.add_argument("--adaptive-policy",type=Path,default=CONFIG_DIR/"adaptive_alpha_sender_local_v2.json")
    parser.add_argument("--integrity-rule",type=Path,default=CONFIG_DIR/"ecc_integrity_rejection_v1.json"); parser.add_argument("--map-offsets",type=Path,default=CONFIG_DIR/"map_time_offsets.json")
    parser.add_argument("--limit-replays",type=int); args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    config,adaptive,panel=load_locked(args); new=run(args,config,adaptive,panel)
    physical=pd.read_csv(args.output_dir/"physical_results.csv"); expected=int(panel.eligible_7_5.sum())*len(config["new_keys"])*len(config["layout_seeds"])
    pd.DataFrame([{"check":"new_configs_this_run","value":new},{"check":"expected_configs","value":expected},{"check":"physical_configs","value":len(physical)},
        {"check":"duplicate_configs","value":int(physical.config_id.duplicated().sum())}]).to_csv(args.output_dir/"diagnostics.csv",index=False)
    print(f"Phase2 physical complete new_configs={new}")


if __name__=="__main__": main()
