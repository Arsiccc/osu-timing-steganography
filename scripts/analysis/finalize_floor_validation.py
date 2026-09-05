"""Post-lock summaries and uncertainty for floor-policy validation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_pilot_ber_sweep import stable_seed


OUTPUT = RESULTS_DIR / "one_codeword_floor_validation_v1"
ITERATIONS = 5000


def ratio_interval(frame: pd.DataFrame, cluster: str, numerator: str,
                   denominator: str | None, seed_tag: str) -> tuple[float, float]:
    grouped = frame.groupby(cluster)
    numerators = grouped[numerator].sum().to_numpy(float)
    denominators = (grouped[denominator].sum().to_numpy(float)
                    if denominator is not None else grouped.size().to_numpy(float))
    rng = np.random.default_rng(stable_seed(42, "floor-validation-final", seed_tag))
    indices = rng.integers(0, len(numerators), size=(ITERATIONS, len(numerators)))
    values = numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    return tuple(float(value) for value in np.quantile(values, [.025, .975]))


def reliability_uncertainty(physical: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    integrity = messages[messages.method == "integrity"]
    rows = []
    for capacity_class, group in [("ALL_ATTEMPTED", physical), *list(physical.groupby("capacity_class"))]:
        merged = group.merge(integrity[["config_id","correct_accept","wrong_accept","reject"]], on="config_id", validate="one_to_one")
        statistics = {
            "raw_ber": ("raw_bit_errors", "coded_bits"),
            "post_ecc_ber": ("post_ecc_bit_errors", "useful_bits"),
            "correct_accept_rate": ("correct_accept", None),
            "wrong_accept_rate": ("wrong_accept", None),
            "reject_rate": ("reject", None),
        }
        for metric, (numerator, denominator) in statistics.items():
            point = float(merged[numerator].sum() / (merged[denominator].sum() if denominator else len(merged)))
            low, high = ratio_interval(merged, "replay_file", numerator, denominator, f"{capacity_class}:{metric}")
            rows.append({"capacity_class":capacity_class,"metric":metric,"value":point,"ci95_low":low,"ci95_high":high,
                         "bootstrap_unit":"replay_file","bootstrap_iterations":ITERATIONS,
                         "configs":len(merged),"replays":merged.replay_file.nunique()})
    return pd.DataFrame(rows)


def system_summary(classification: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    integrity = messages[messages.method == "integrity"]
    total = len(classification); denominator = total * 5
    counts = classification.capacity_class.value_counts()
    correct = int(integrity.correct_accept.sum()); wrong = int(integrity.wrong_accept.sum()); reject = int(integrity.reject.sum())
    no_capacity = int(counts.get("NO_CAPACITY", 0)); useful = int(integrity.useful_correctly_accepted_bits.sum())
    map_class = classification[["beatmap_hash","capacity_class"]].drop_duplicates()
    replay_seed = pd.MultiIndex.from_product([classification.replay_file.astype(str), range(5)], names=["replay_file","layout_seed"]).to_frame(index=False)
    outcomes = replay_seed.merge(classification[["replay_file","beatmap_hash","capacity_class","useful_bits"]],on="replay_file",validate="many_to_one")
    outcomes = outcomes.merge(integrity[["replay_file","layout_seed","correct_accept","wrong_accept","reject","useful_correctly_accepted_bits"]],
                              on=["replay_file","layout_seed"],how="left",validate="one_to_one").fillna(
                                  {"correct_accept":0,"wrong_accept":0,"reject":0,"useful_correctly_accepted_bits":0})
    rows = []
    metrics = {
        "normal_coverage": (int(counts.get("NORMAL_7_5",0)), total, counts.get("NORMAL_7_5",0)/total, None),
        "floor_coverage_gain": (int(counts.get("FLOOR_12_5",0)), total, counts.get("FLOOR_12_5",0)/total, None),
        "total_attempt_coverage": (len(classification)-no_capacity,total,(len(classification)-no_capacity)/total,None),
        "no_capacity_abstention": (no_capacity,total,no_capacity/total,None),
        "correct_delivery": (correct,denominator,correct/denominator,"correct_accept"),
        "silent_wrong_delivery": (wrong,denominator,wrong/denominator,"wrong_accept"),
        "explicit_decoder_rejection": (reject,denominator,reject/denominator,"reject"),
        "correctly_accepted_useful_information_per_input_replay": (useful,denominator,useful/denominator,"useful_correctly_accepted_bits"),
    }
    for metric,(numerator,denom,value,column) in metrics.items():
        if column is None:
            per_map = classification.groupby("beatmap_hash").capacity_class.apply(
                lambda values: {"normal_coverage":(values=="NORMAL_7_5").mean(),"floor_coverage_gain":(values=="FLOOR_12_5").mean(),
                                "total_attempt_coverage":(values!="NO_CAPACITY").mean(),"no_capacity_abstention":(values=="NO_CAPACITY").mean()}[metric])
            rng=np.random.default_rng(stable_seed(42,"floor-validation-system",metric)); array=per_map.to_numpy(float)
            samples=array[rng.integers(0,len(array),size=(ITERATIONS,len(array)))].mean(axis=1); low,high=np.quantile(samples,[.025,.975])
        else:
            low,high=ratio_interval(outcomes,"beatmap_hash",column,None,f"system:{metric}")
        rows.append({"metric":metric,"numerator":numerator,"denominator":denom,"value":value,"ci95_low":low,"ci95_high":high,
                     "bootstrap_unit":"beatmap_hash","bootstrap_iterations":ITERATIONS})
    return pd.DataFrame(rows)


def cost_benefit(classification: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    integrity=messages[messages.method=="integrity"]
    total=len(classification)*5
    normal=integrity[integrity.capacity_class=="NORMAL_7_5"]
    floor=integrity[integrity.capacity_class=="FLOOR_12_5"]
    base={"coverage":(classification.capacity_class=="NORMAL_7_5").mean(),"correct_delivery":normal.correct_accept.sum()/total,
          "silent_wrong_delivery":normal.wrong_accept.sum()/total,"explicit_rejection":normal.reject.sum()/total,
          "abstention":(classification.capacity_class!="NORMAL_7_5").mean(),
          "useful_information_per_input_replay":normal.useful_correctly_accepted_bits.sum()/total}
    added={"coverage":(classification.capacity_class!="NO_CAPACITY").mean(),"correct_delivery":integrity.correct_accept.sum()/total,
           "silent_wrong_delivery":integrity.wrong_accept.sum()/total,"explicit_rejection":integrity.reject.sum()/total,
           "abstention":(classification.capacity_class=="NO_CAPACITY").mean(),
           "useful_information_per_input_replay":integrity.useful_correctly_accepted_bits.sum()/total}
    return pd.DataFrame([{"metric":key,"base_7_5":base[key],"with_floor_12_5":added[key],"delta_floor_minus_base":added[key]-base[key]} for key in base])


def beatmap_summary(classification: pd.DataFrame, physical: pd.DataFrame, messages: pd.DataFrame,
                    features: pd.DataFrame, detector_map: pd.DataFrame) -> pd.DataFrame:
    integrity=messages[messages.method=="integrity"]
    clean=features[features.label==0].drop_duplicates("replay_file")
    rows=[]
    for beatmap, group in classification.groupby("beatmap_hash"):
        capacity_class=str(group.capacity_class.iloc[0]); p=physical[physical.beatmap_hash==beatmap]; m=integrity[integrity.beatmap_hash==beatmap]
        auc=detector_map[(detector_map.beatmap_hash==beatmap)]
        rows.append({"beatmap_hash":beatmap,"capacity_class":capacity_class,"input_replays":len(group),
                     "normal_replays":int((group.capacity_class=="NORMAL_7_5").sum()),"floor_replays":int((group.capacity_class=="FLOOR_12_5").sum()),
                     "no_capacity_replays":int((group.capacity_class=="NO_CAPACITY").sum()),"configs":len(p),
                     "raw_ber":p.raw_bit_errors.sum()/p.coded_bits.sum() if len(p) else np.nan,
                     "correct_accept_rate":m.correct_accept.mean() if len(m) else np.nan,"wrong_accept_rate":m.wrong_accept.mean() if len(m) else np.nan,
                     "reject_rate":m.reject.mean() if len(m) else np.nan,
                     "clean_valid_residual_fraction_mean":clean.loc[clean.beatmap_hash==beatmap,"valid_residual_fraction"].mean(),
                     "clean_valid_residual_fraction_min":clean.loc[clean.beatmap_hash==beatmap,"valid_residual_fraction"].min(),
                     "full_auc_seed_mean":auc.full_auc.mean() if len(auc) else np.nan,"full_auc_seed_min":auc.full_auc.min() if len(auc) else np.nan,
                     "full_auc_seed_max":auc.full_auc.max() if len(auc) else np.nan})
    return pd.DataFrame(rows)


def zero_event_bounds(messages: pd.DataFrame) -> pd.DataFrame:
    integrity=messages[messages.method=="integrity"]
    rows=[]
    for capacity_class,group in [("ALL_ATTEMPTED",integrity),*list(integrity.groupby("capacity_class"))]:
        events=int(group.wrong_accept.sum()); configs=len(group); replays=group.replay_file.nunique()
        rows.extend([{"capacity_class":capacity_class,"unit":"configuration_independence_naive","events":events,"units":configs,
                      "one_sided_95pct_upper":1-.05**(1/configs) if events==0 else np.nan},
                     {"capacity_class":capacity_class,"unit":"replay_any_seed_conservative","events":group.loc[group.wrong_accept==1,"replay_file"].nunique(),
                      "units":replays,"one_sided_95pct_upper":1-.05**(1/replays) if events==0 else np.nan}])
    return pd.DataFrame(rows)


def integrity_message_analysis(messages: pd.DataFrame, words: pd.DataFrame, output: Path) -> None:
    wide = messages.pivot(index=["config_id","replay_file","beatmap_hash","capacity_class","layout_seed"],
                          columns="method", values="outcome").reset_index()
    wide.groupby(["capacity_class","baseline","integrity"]).size().rename("configs").reset_index().to_csv(
        output/"integrity_message_transitions.csv",index=False)
    rows=[]
    for capacity_class,group in wide.groupby("capacity_class"):
        baseline_wrong=group.baseline.eq("WRONG_ACCEPT"); caught=baseline_wrong & group.integrity.eq("REJECT")
        baseline_correct=group.baseline.eq("CORRECT_ACCEPT"); false_reject=baseline_correct & group.integrity.eq("REJECT")
        rows.append({"capacity_class":capacity_class,"configs":len(group),
                     "baseline_wrong_accept_messages":int(baseline_wrong.sum()),"silent_messages_captured":int(caught.sum()),
                     "silent_message_capture_rate":caught.sum()/baseline_wrong.sum() if baseline_wrong.any() else np.nan,
                     "baseline_correct_accept_messages":int(baseline_correct.sum()),"additional_correct_message_rejects":int(false_reject.sum()),
                     "message_false_rejection_rate":false_reject.sum()/baseline_correct.sum() if baseline_correct.any() else np.nan})
    pd.DataFrame(rows).to_csv(output/"integrity_message_summary.csv",index=False)
    words.groupby(["capacity_class","integrity_reason"]).size().rename("codewords").reset_index().to_csv(
        output/"integrity_rejection_reasons.csv",index=False)
    words.dropna(subset=["corrected_bit_confidence_rank"]).groupby(
        ["capacity_class","word_ground_truth_correct"])["corrected_bit_confidence_rank"].agg(
            ["count","mean","median","min","max"]).reset_index().to_csv(output/"corrected_rank_diagnostics.csv",index=False)


def report(output: Path, classification: pd.DataFrame, reliability: pd.DataFrame, system: pd.DataFrame,
           integrity: pd.DataFrame, physical: pd.DataFrame, detector: pd.DataFrame, beatmaps: pd.DataFrame) -> None:
    rel=reliability.pivot(index="capacity_class",columns="metric",values="value"); sys=system.set_index("metric"); integ=integrity.set_index("capacity_class")
    full=detector[detector.feature_set=="full"].groupby("capacity_class").roc_auc.agg(["mean","min","max"])
    floor_bad=beatmaps[beatmaps.capacity_class=="FLOOR_12_5"].sort_values("raw_ber",ascending=False).iloc[0]
    text=f"""# Frozen 12.5%-capped one-codeword-floor validation

## Pre-registration and corpus

Policy SHA-256: `92d9224974838a0628aee9df0c3d112b6c86fb981fe057a3c31acb2daf22ac78`. Validation-lock SHA-256: `a9715152f19d8b6f52f2aef85dd60581ebcf1a59eb755ebc5fcc722cbf1800e3`. The primary corpus contains 300 replay files on 20 new beatmaps, with zero historical beatmap, replay-ID, and normalized-username overlap. Map offsets came from a disjoint 240-replay calibration corpus.

## Coverage and system outcome

NORMAL_7_5: 135/300 (45%); FLOOR_12_5: 45/300 (15% gain); NO_CAPACITY: 120/300 (40%); total attempt coverage: 60%. Across 300 input replays and the five frozen layout keys, correct delivery was {sys.loc['correct_delivery','numerator']:.0f}/{sys.loc['correct_delivery','denominator']:.0f} ({sys.loc['correct_delivery','value']:.2%}), silent wrong delivery {sys.loc['silent_wrong_delivery','numerator']:.0f}/{sys.loc['silent_wrong_delivery','denominator']:.0f}, explicit rejection {sys.loc['explicit_decoder_rejection','numerator']:.0f}/{sys.loc['explicit_decoder_rejection','denominator']:.0f} ({sys.loc['explicit_decoder_rejection','value']:.2%}), and correctly accepted useful information was {sys.loc['correctly_accepted_useful_information_per_input_replay','value']:.3f} bits/input replay.

## Reliability and integrity

NORMAL raw BER {rel.loc['NORMAL_7_5','raw_ber']:.2%}, CORRECT_ACCEPT {rel.loc['NORMAL_7_5','correct_accept_rate']:.2%}, REJECT {rel.loc['NORMAL_7_5','reject_rate']:.2%}. FLOOR raw BER {rel.loc['FLOOR_12_5','raw_ber']:.2%}, CORRECT_ACCEPT {rel.loc['FLOOR_12_5','correct_accept_rate']:.2%}, REJECT {rel.loc['FLOOR_12_5','reject_rate']:.2%}. Frozen integrity produced zero wrong accepted messages. It captured {int(integ.loc['FLOOR_12_5','silent_captured'])}/{int(integ.loc['FLOOR_12_5','baseline_wrong_accept_words'])} floor and {int(integ.loc['NORMAL_7_5','silent_captured'])}/{int(integ.loc['NORMAL_7_5','baseline_wrong_accept_words'])} normal wrong-accepted codewords, with word-level additional false-rejection rates {integ.loc['FLOOR_12_5','false_rejection_rate']:.2%} and {integ.loc['NORMAL_7_5','false_rejection_rate']:.2%} respectively.

## Detectability and heterogeneity

Frozen development-to-new-map FULL AUC across seeds: ALL {full.loc['ALL_ATTEMPTED','mean']:.3f} ({full.loc['ALL_ATTEMPTED','min']:.3f}–{full.loc['ALL_ATTEMPTED','max']:.3f}), NORMAL {full.loc['NORMAL_7_5','mean']:.3f}, FLOOR {full.loc['FLOOR_12_5','mean']:.3f}. Floor detectability did not worsen relative to the exploratory pilot, but floor reliability did: pilot raw BER 3.94% / correct 90.0% / reject 9.75%, versus new-map 5.89% / 78.67% / 21.33%. The worst locked floor map (`{floor_bad.beatmap_hash}`) had raw BER {floor_bad.raw_ber:.2%}, correct accept {floor_bad.correct_accept_rate:.2%}, and minimum clean valid-residual fraction {floor_bad.clean_valid_residual_fraction_min:.2%}; it was retained as required.

## Decision

**C — reject the floor as a default validated system policy; keep the frozen 7.5% abstaining policy.** The floor gains 15 percentage points of attempt coverage and 11.8 points of correct delivery, but newly covered replays show materially worse and map-heterogeneous reliability. Zero observed silent accepts is encouraging but does not establish zero risk. A future optional-mode study would require a new preregistered corpus; this validation does not authorize retuning on these data.
"""
    (output/"validation_report.md").write_text(text)


def hash_manifest(output: Path) -> None:
    paths=sorted(path for path in output.iterdir() if path.is_file() and path.name!="result_hashes.json")
    data={"hash_algorithm":"sha256","files":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (output/"result_hashes.json").write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


def main() -> None:
    classification=pd.read_csv(OUTPUT/"capacity_classification.csv"); physical=pd.read_csv(OUTPUT/"physical_results.csv")
    messages=pd.read_csv(OUTPUT/"message_results.csv"); words=pd.read_csv(OUTPUT/"codeword_results.csv")
    features=pd.read_csv(OUTPUT/"timing_features.csv"); detector=pd.read_csv(OUTPUT/"detector_auc.csv")
    detector_map=pd.read_csv(OUTPUT/"detector_by_map.csv")
    reliability=reliability_uncertainty(physical,messages); reliability.to_csv(OUTPUT/"reliability_uncertainty.csv",index=False)
    system=system_summary(classification,messages); system.to_csv(OUTPUT/"system_metrics_summary.csv",index=False)
    cost_benefit(classification,messages).to_csv(OUTPUT/"policy_cost_benefit.csv",index=False)
    maps=beatmap_summary(classification,physical,messages,features,detector_map); maps.to_csv(OUTPUT/"beatmap_summary.csv",index=False)
    zero_event_bounds(messages).to_csv(OUTPUT/"zero_event_bounds.csv",index=False)
    integrity_message_analysis(messages,words,OUTPUT)
    integrity=pd.read_csv(OUTPUT/"integrity_by_class.csv"); report(OUTPUT,classification,reliability,system,integrity,physical,detector,maps)
    hash_manifest(OUTPUT)


if __name__=="__main__":
    main()
