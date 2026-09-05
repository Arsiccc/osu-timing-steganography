"""Focused pre-outcome sanity checks for PN robustness Phase 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.stego.payload_layout import embed_message_with_layout, extract_bipolar_message_with_layout, select_payload_blocks
from osu_stego.stego.pn_sequence import _derive_seed, generate_pn_sequence
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS
from scripts.experiments.run_pn_key_robustness_phase2 import identity


OUTPUT=Path("results/pn_key_robustness_phase2_v1")


def main() -> None:
    config=json.loads(Path("data/config/pn_key_robustness_phase2_v1.json").read_text()); phase1=json.loads(Path("data/config/pn_key_robustness_v1.json").read_text())
    new=config["new_keys"]; prior=[phase1["historical_reference"],*phase1["new_keys"]]
    assert len(new)==30 and len({x["key"] for x in new})==30 and len({x["key_id"] for x in new})==30
    seeds=[_derive_seed(x["key"]) for x in [*prior,*new]]; assert len(seeds)==len(set(seeds))==41
    residuals=np.zeros(1024); message=np.asarray([-1,1,1,-1,1,-1,-1,1],dtype=np.int8)
    blocks=select_payload_blocks(1024,8,len(message),LAYOUT_KEYS[0],"distributed")
    ids=[]
    for item in new:
        a=generate_pn_sequence(item["key"],64); b=generate_pn_sequence(item["key"],64); assert np.array_equal(a,b)
        stego=embed_message_with_layout(residuals,message,item["key"],LAYOUT_KEYS[0],12,8,"distributed")
        decoded=extract_bipolar_message_with_layout(stego,item["key"],LAYOUT_KEYS[0],8,len(message),"distributed"); assert np.array_equal(decoded,message)
        assert np.array_equal(blocks,select_payload_blocks(1024,8,len(message),LAYOUT_KEYS[0],"distributed"))
        ids.append(identity("replay.osr","sha",item["key_id"],0,message))
    assert len(ids)==len(set(ids))
    panel=pd.read_csv(OUTPUT/"replay_panel.csv"); assert len(panel)==100 and int(panel.eligible_7_5.sum())==68
    reproduction=pd.read_csv("results/pn_key_robustness_v1/historical_reproduction.csv")
    assert (reproduction.matching_configs==reproduction.configs).all()
    collisions=pd.read_csv(OUTPUT/"effective_seed_collisions.csv"); assert collisions.empty
    print("PN PHASE2 CHECK OK: keys=30 effective_seeds=41 distinct noiseless_ber=0")


if __name__=="__main__": main()
