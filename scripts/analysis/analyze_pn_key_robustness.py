"""Analyze the locked multi-PN physical pilot without tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.timing_features import BASELINE_FEATURES, FULL_FEATURES, POSITION_FEATURES
from osu_stego.paths import RESULTS_DIR
from osu_stego.stego.pn_sequence import generate_pn_sequence
from scripts.experiments.run_adaptive_layout_strong import LEGACY_STRONG_EXPERIMENT
from scripts.experiments.run_pilot_ber_sweep import stable_seed


OUTPUT = RESULTS_DIR / "pn_key_robustness_v1"
FEATURE_SETS = {"baseline": tuple(BASELINE_FEATURES), "position_only": tuple(POSITION_FEATURES), "full": tuple(FULL_FEATURES)}
BOOTSTRAPS = 2000


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, n_jobs=1,
        random_state=stable_seed(42, LEGACY_STRONG_EXPERIMENT, "generalization-rf"))


def clustered_delta(frame: pd.DataFrame, key_id: str, metric: str, seed: int) -> tuple[float, float, float]:
    historical = frame[frame.historical_pn == 1].set_index(["replay_file", "layout_seed"])
    current = frame[frame.pn_key_id == key_id].set_index(["replay_file", "layout_seed"])
    paired = current[[metric]].join(historical[[metric]], lsuffix="_new", rsuffix="_historical", validate="one_to_one").reset_index()
    by_replay = paired.groupby("replay_file")[[f"{metric}_new", f"{metric}_historical"]].mean()
    delta = by_replay.iloc[:, 0] - by_replay.iloc[:, 1]
    rng = np.random.default_rng(seed); values = delta.to_numpy(); n = len(values)
    samples = np.asarray([np.mean(values[rng.integers(0, n, n)]) for _ in range(BOOTSTRAPS)])
    return float(delta.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def reliability(physical: pd.DataFrame, messages: pd.DataFrame) -> None:
    rows = []
    merged = physical.merge(messages[["config_id", "outcome", "post_ecc_bit_errors", "post_ecc_ber"]], on="config_id", validate="one_to_one")
    for pn, g in merged.groupby("pn_key_id", sort=False):
        rows.append({"pn_key_id": pn, "historical_pn": int(g.historical_pn.iloc[0]), "configs": len(g), "replays": g.replay_file.nunique(),
            "raw_bit_errors": int(g.raw_bit_errors.sum()), "coded_bits": int(g.coded_bits.sum()), "weighted_raw_ber": g.raw_bit_errors.sum()/g.coded_bits.sum(),
            "replay_median_ber": g.groupby("replay_file").apply(lambda x: x.raw_bit_errors.sum()/x.coded_bits.sum(), include_groups=False).median(),
            "post_ecc_bit_errors": int(g.post_ecc_bit_errors.sum()), "useful_bits": int(g.useful_bits.sum()), "post_ecc_ber": g.post_ecc_bit_errors.sum()/g.useful_bits.sum(),
            "correct_accept": int((g.outcome == "CORRECT_ACCEPT").sum()), "wrong_accept": int((g.outcome == "WRONG_ACCEPT").sum()),
            "reject": int((g.outcome == "REJECT").sum()), "correct_accept_rate": (g.outcome == "CORRECT_ACCEPT").mean(),
            "wrong_accept_rate": (g.outcome == "WRONG_ACCEPT").mean(), "reject_rate": (g.outcome == "REJECT").mean()})
    pd.DataFrame(rows).to_csv(OUTPUT / "reliability_by_pn.csv", index=False)
    paired = []
    hist_id = str(merged.loc[merged.historical_pn == 1, "pn_key_id"].iloc[0])
    for pn in merged.pn_key_id.unique():
        if pn == hist_id: continue
        for metric in ("raw_ber", "post_ecc_ber", "correct_accept", "wrong_accept", "reject"):
            source = merged.copy()
            if metric in ("correct_accept", "wrong_accept", "reject"):
                source[metric] = (source.outcome == metric.upper()).astype(float)
            delta, low, high = clustered_delta(source, str(pn), metric, stable_seed(42, pn, metric, "paired-bootstrap"))
            paired.append({"pn_key_id": pn, "reference_pn_key_id": hist_id, "metric": metric, "paired_delta": delta,
                           "ci95_low": low, "ci95_high": high, "bootstrap_unit": "replay", "iterations": BOOTSTRAPS})
    pd.DataFrame(paired).to_csv(OUTPUT / "reliability_paired.csv", index=False)
    pd.DataFrame(rows)[["pn_key_id","historical_pn","configs","correct_accept","wrong_accept","reject","correct_accept_rate","wrong_accept_rate","reject_rate"]].to_csv(OUTPUT / "integrity_by_pn.csv", index=False)


def physical_summary(physical: pd.DataFrame, messages: pd.DataFrame) -> None:
    columns = ["active_carrier_fraction", "requested_carriers", "active_carriers", "sum_squared_shift_ms2", "rms_applied_shift_ms",
               "mean_absolute_applied_shift_ms", "dropped_for_chronology", "dropped_for_hit_window", "changed_match_status_total"]
    physical.groupby(["pn_key_id", "historical_pn"])[columns].mean().reset_index().to_csv(OUTPUT / "physical_by_pn.csv", index=False)
    matrix = physical.merge(messages[["config_id", "correct_accept", "reject"]], on="config_id", validate="one_to_one")
    matrix.groupby(["pn_key_id", "historical_pn", "layout_seed"]).agg(raw_bit_errors=("raw_bit_errors","sum"), coded_bits=("coded_bits","sum"),
        raw_ber=("raw_ber","mean"), correct_accept_rate=("correct_accept","mean"), reject_rate=("reject","mean"),
        chronology_drops=("dropped_for_chronology","sum")).reset_index().to_csv(OUTPUT / "pn_layout_matrix.csv", index=False)


def confidence(bits: pd.DataFrame, words: pd.DataFrame) -> None:
    rows=[]
    for pn,g in bits.groupby("pn_key_id", sort=False):
        errors=g.is_error.astype(int); auc=float(roc_auc_score(errors,-g.abs_correlation)) if errors.nunique()==2 else np.nan
        w=words[words.pn_key_id==pn]
        rows.append({"pn_key_id":pn,"historical_pn":int(g.historical_pn.iloc[0]),"bits":len(g),"errors":int(errors.sum()),
            "median_abs_c_correct":g.loc[errors==0,"abs_correlation"].median(),"median_abs_c_error":g.loc[errors==1,"abs_correlation"].median(),
            "low_abs_c_error_auc":auc,"corrected_words":int((w.hard_status=="corrected_single").sum()),
            "corrected_rank_mean":w.corrected_bit_confidence_rank.mean(),"wrong_accepted_words":int(((w.word_correct==0)&(w.integrity_accepted==1)).sum())})
    pd.DataFrame(rows).to_csv(OUTPUT / "confidence_by_pn.csv",index=False)


def cv_auc(frame: pd.DataFrame, names: tuple[str,...], group_col: str) -> float:
    scores=np.zeros(len(frame)); groups=frame[group_col].astype(str).to_numpy(); splitter=GroupKFold(5)
    for train,test in splitter.split(frame,frame.label,groups):
        model=rf(); model.fit(frame.iloc[train][list(names)],frame.iloc[train].label); scores[test]=model.predict_proba(frame.iloc[test][list(names)])[:,1]
    return float(roc_auc_score(frame.label,scores))


def detectors(features: pd.DataFrame, selection: pd.DataFrame) -> None:
    old=pd.read_csv(RESULTS_DIR/"adaptive_layout_strong_v1"/"features.csv")
    pilot=set(selection.loc[selection.eligible_7_5==1,"replay_file"].astype(str))
    old=old[(old.partition=="development")&(old.alpha_method=="sender_local_adaptive")&(old.layout=="distributed")&~old.replay_file.astype(str).isin(pilot)]
    frozen=[]; aware=[]
    for (pn,seed),test in features.groupby(["pn_key_id","layout_seed"],sort=False):
        train=old[old.layout_seed.astype(str)==str(seed)]
        for label,names in FEATURE_SETS.items():
            model=rf(); model.fit(train[list(names)],train.label); score=model.predict_proba(test[list(names)])[:,1]
            frozen.append({"pn_key_id":pn,"layout_seed":seed,"feature_set":label,"roc_auc":roc_auc_score(test.label,score),
                           "train_replays":train.replay_file.nunique(),"test_replays":test.replay_file.nunique(),"replay_overlap":len(set(train.replay_file)&set(test.replay_file))})
            aware.append({"pn_key_id":pn,"layout_seed":seed,"feature_set":label,"roc_auc":cv_auc(test.reset_index(drop=True),names,"replay_file"),
                          "cv_group":"replay_file","folds":5})
    pd.DataFrame(frozen).to_csv(OUTPUT/"detector_frozen_by_pn.csv",index=False)
    pd.DataFrame(aware).to_csv(OUTPUT/"detector_keyaware_by_pn.csv",index=False)

    multi=[]
    maps=np.array(sorted(features.beatmap_hash.astype(str).unique())); fold_by_map={m:i%5 for i,m in enumerate(maps)}
    for holdout in features.pn_key_id.unique():
        for label,names in FEATURE_SETS.items():
            labels=[]; scores=[]
            for fold in range(5):
                test=features[(features.pn_key_id==holdout)&(features.beatmap_hash.astype(str).map(fold_by_map)==fold)]
                train=features[(features.pn_key_id!=holdout)&(features.beatmap_hash.astype(str).map(fold_by_map)!=fold)]
                model=rf(); model.fit(train[list(names)],train.label); labels.extend(test.label); scores.extend(model.predict_proba(test[list(names)])[:,1])
            multi.append({"heldout_pn_key_id":holdout,"feature_set":label,"roc_auc":roc_auc_score(labels,scores),"split":"heldout_pn_plus_disjoint_beatmap_fold","folds":5})
    pd.DataFrame(multi).to_csv(OUTPUT/"detector_multikey_by_pn.csv",index=False)


def feature_deltas(features: pd.DataFrame) -> None:
    names=["variance","excess_kurtosis","autocorrelation_lag1","mean_absolute_first_difference","variance_first_difference",
           *[x for x in FULL_FEATURES if x.startswith("quarter_")]]
    clean=features[features.label==0].set_index("config_id"); stego=features[features.label==1].set_index("config_id")
    rows=[]
    for pn,ids in stego.groupby("pn_key_id").groups.items():
        for name in names: rows.append({"pn_key_id":pn,"feature":name,"mean_stego_minus_clean":float((stego.loc[ids,name]-clean.loc[ids,name]).mean())})
    pd.DataFrame(rows).to_csv(OUTPUT/"feature_deltas_by_pn.csv",index=False)


def pn_diagnostics(config: dict, physical: pd.DataFrame) -> None:
    length=int(physical.coded_bits.max()*8); rows=[]
    keys=[config["historical_reference"],*config["new_keys"]]
    for item in keys:
        x=generate_pn_sequence(item["key"],length).astype(float); runs=[]; start=0
        for i in range(1,len(x)+1):
            if i==len(x) or x[i]!=x[start]: runs.append(i-start); start=i
        row={"pn_key_id":item["key_id"],"chips":len(x),"fraction_positive":float(np.mean(x>0)),"mean_run_length":float(np.mean(runs)),"max_run_length":max(runs)}
        for lag in (1,2,4,8): row[f"autocorrelation_lag{lag}"]=float(np.corrcoef(x[:-lag],x[lag:])[0,1])
        blocks=x.reshape(-1,8); row["block_positive_fraction_sd"]=float(np.std(np.mean(blocks>0,axis=1))); rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT/"pn_sequence_diagnostics.csv",index=False)


def finalize() -> None:
    physical=pd.read_csv(OUTPUT/"physical_results.csv"); messages=pd.read_csv(OUTPUT/"message_results.csv"); bits=pd.read_csv(OUTPUT/"bit_results.csv")
    words=pd.read_csv(OUTPUT/"codeword_results.csv"); features=pd.read_csv(OUTPUT/"timing_features.csv"); selection=pd.read_csv(OUTPUT/"pilot_selection.csv")
    config=json.loads((OUTPUT/"pn_keys.json").read_text())
    reliability(physical,messages); physical_summary(physical,messages); confidence(bits,words); detectors(features,selection); feature_deltas(features); pn_diagnostics(config,physical)
    wrong=pd.read_csv(OUTPUT/"wrong_key_sanity.csv").groupby("pn_key_id").agg(configs=("config_id","size"),wrong_key_ber=("wrong_key_ber","mean"),mean_abs_correlation=("wrong_key_mean_abs_correlation","mean")).reset_index()
    wrong.to_csv(OUTPUT/"wrong_key_sanity_summary.csv",index=False)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!="result_hashes.json"}
    (OUTPUT/"result_hashes.json").write_text(json.dumps({"hash_algorithm":"sha256","files":files},indent=2,sort_keys=True)+"\n")
    print("PN robustness analysis complete")


if __name__=="__main__": finalize()
