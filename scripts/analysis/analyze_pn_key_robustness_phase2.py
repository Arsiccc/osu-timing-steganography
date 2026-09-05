"""Frozen analysis for PN-key robustness Phase 2."""

from __future__ import annotations

import hashlib
import json
import math
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


OUTPUT=RESULTS_DIR/"pn_key_robustness_phase2_v1"; PHASE1=RESULTS_DIR/"pn_key_robustness_v1"; ITERATIONS=2000
FEATURE_SETS={"baseline":tuple(BASELINE_FEATURES),"position_only":tuple(POSITION_FEATURES),"full":tuple(FULL_FEATURES)}


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300,max_features="sqrt",min_samples_leaf=2,n_jobs=1,
        random_state=stable_seed(42,LEGACY_STRONG_EXPERIMENT,"generalization-rf"))


def summarize_keys(physical: pd.DataFrame,messages: pd.DataFrame) -> pd.DataFrame:
    joined=physical.merge(messages[["config_id","post_ecc_bit_errors","outcome","correct_accept","wrong_accept","reject"]],on="config_id",validate="one_to_one")
    rows=[]
    for pn,g in joined.groupby("pn_key_id",sort=False):
        replay=g.groupby("replay_file").agg(errors=("raw_bit_errors","sum"),bits=("coded_bits","sum")); replay["ber"]=replay.errors/replay.bits
        rows.append({"pn_key_id":pn,"pn_key_index":int(g.pn_key_index.iloc[0]),"configs":len(g),"replays":g.replay_file.nunique(),
            "raw_errors":int(g.raw_bit_errors.sum()),"coded_bits":int(g.coded_bits.sum()),"weighted_raw_ber":g.raw_bit_errors.sum()/g.coded_bits.sum(),
            "mean_replay_ber":replay.ber.mean(),"median_replay_ber":replay.ber.median(),"post_ecc_errors":int(g.post_ecc_bit_errors.sum()),
            "useful_bits":int(g.useful_bits.sum()),"post_ecc_ber":g.post_ecc_bit_errors.sum()/g.useful_bits.sum(),
            "correct_accept":int(g.correct_accept.sum()),"wrong_accept":int(g.wrong_accept.sum()),"reject":int(g.reject.sum()),
            "correct_accept_rate":g.correct_accept.mean(),"wrong_accept_rate":g.wrong_accept.mean(),"reject_rate":g.reject.mean()})
    return pd.DataFrame(rows)


def distribution(reliability: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for metric in ("weighted_raw_ber","post_ecc_ber","correct_accept_rate","wrong_accept_rate","reject_rate"):
        x=reliability[metric]
        row={"metric":metric,"keys":len(x),"mean":x.mean(),"sd":x.std(ddof=1),"median":x.median(),"min":x.min(),"max":x.max()}
        for q in (.1,.25,.75,.9): row[f"q{int(q*100)}"]=x.quantile(q)
        rows.append(row)
    return pd.DataFrame(rows)


def common_design(phase2: pd.DataFrame,phase2_messages: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    p1=pd.read_csv(PHASE1/"physical_results.csv"); m1=pd.read_csv(PHASE1/"message_results.csv")
    p1=p1[p1.layout_seed<3].copy(); p1["phase"]="phase1"; phase2=phase2.copy(); phase2["phase"]="phase2"
    combined=pd.concat([p1,phase2],ignore_index=True,sort=False)
    m1=m1[m1.config_id.isin(set(p1.config_id))].copy(); m1["phase"]="phase1"; phase2_messages=phase2_messages.copy(); phase2_messages["phase"]="phase2"
    messages=pd.concat([m1,phase2_messages],ignore_index=True,sort=False)
    summary=summarize_keys(combined,messages); phase={**dict(zip(p1.pn_key_id.unique(),["phase1"]*p1.pn_key_id.nunique())),**dict(zip(phase2.pn_key_id.unique(),["phase2"]*phase2.pn_key_id.nunique()))}
    summary["phase"]=summary.pn_key_id.map(phase); hist=str(p1.loc[p1.historical_pn==1,"pn_key_id"].iloc[0]); summary.loc[summary.pn_key_id==hist,"phase"]="historical"
    return combined,messages,summary


def bootstrap_mean(physical: pd.DataFrame,messages: pd.DataFrame) -> pd.DataFrame:
    joined=physical.merge(messages[["config_id","correct_accept","reject"]],on="config_id",validate="one_to_one")
    keys=np.array(sorted(joined.pn_key_id.unique())); replays=np.array(sorted(joined.replay_file.unique())); index=pd.MultiIndex.from_product([keys,replays])
    cell=joined.groupby(["pn_key_id","replay_file"]).agg(errors=("raw_bit_errors","sum"),bits=("coded_bits","sum"),correct=("correct_accept","sum"),reject=("reject","sum"),configs=("config_id","size")).reindex(index)
    arrays={c:cell[c].to_numpy().reshape(len(keys),len(replays)) for c in cell.columns}; rng=np.random.default_rng(20260902); values={m:[] for m in ("mean_raw_ber","mean_correct_accept","mean_reject")}
    for _ in range(ITERATIONS):
        ki=rng.integers(0,len(keys),len(keys)); ri=rng.integers(0,len(replays),len(replays))
        values["mean_raw_ber"].append(arrays["errors"][ki][:,ri].sum()/arrays["bits"][ki][:,ri].sum())
        values["mean_correct_accept"].append(arrays["correct"][ki][:,ri].sum()/arrays["configs"][ki][:,ri].sum())
        values["mean_reject"].append(arrays["reject"][ki][:,ri].sum()/arrays["configs"][ki][:,ri].sum())
    return pd.DataFrame([{"metric":m,"estimate":np.mean(v),"ci95_low":np.quantile(v,.025),"ci95_high":np.quantile(v,.975),
        "bootstrap":"resample PN keys and replay clusters independently; retain all layouts in each sampled cell","iterations":ITERATIONS} for m,v in values.items()])


def replay_uncertainty(physical: pd.DataFrame) -> pd.DataFrame:
    rng=np.random.default_rng(20260903); rows=[]
    for pn,g in physical.groupby("pn_key_id"):
        by=g.groupby("replay_file").agg(errors=("raw_bit_errors","sum"),bits=("coded_bits","sum")); vals=[]
        for _ in range(1000):
            sample=by.iloc[rng.integers(0,len(by),len(by))]; vals.append(sample.errors.sum()/sample.bits.sum())
        rows.append({"pn_key_id":pn,"weighted_raw_ber":g.raw_bit_errors.sum()/g.coded_bits.sum(),"replay_bootstrap_se":np.std(vals,ddof=1),
            "ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975),"iterations":1000})
    return pd.DataFrame(rows)


def layout_analysis(physical: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    matrix=physical.groupby(["pn_key_id","layout_seed"]).agg(raw_errors=("raw_bit_errors","sum"),coded_bits=("coded_bits","sum")); matrix["weighted_raw_ber"]=matrix.raw_errors/matrix.coded_bits
    wide=matrix.weighted_raw_ber.unstack(); pearson=wide.corr(); spearman=wide.corr(method="spearman"); rows=[]
    for a in wide.columns:
        for b in wide.columns: rows.append({"layout_a":a,"layout_b":b,"pearson":pearson.loc[a,b],"spearman":spearman.loc[a,b]})
    out=matrix.reset_index().merge(wide.std(axis=1,ddof=1).rename("per_key_layout_sd"),on="pn_key_id")
    return out,pd.DataFrame(rows)


def confidence(bits: pd.DataFrame,words: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for pn,g in bits.groupby("pn_key_id"):
        errors=g.is_error.astype(int); threshold=g.abs_correlation.quantile(.1); w=words[words.pn_key_id==pn]
        rows.append({"pn_key_id":pn,"bits":len(g),"errors":int(errors.sum()),"median_abs_c_correct":g.loc[errors==0,"abs_correlation"].median(),
            "median_abs_c_error":g.loc[errors==1,"abs_correlation"].median(),"low_abs_c_error_auc":roc_auc_score(errors,-g.abs_correlation),
            "lowest_decile_threshold":threshold,"lowest_decile_error_rate":g.loc[g.abs_correlation<=threshold,"is_error"].mean(),
            "corrected_words":int((w.hard_status=="corrected_single").sum()),"corrected_rank_mean":w.corrected_bit_confidence_rank.mean()})
    return pd.DataFrame(rows)


def pn_diagnostics(config: dict,physical: pd.DataFrame) -> pd.DataFrame:
    length=int(physical.coded_bits.max()*8); rows=[]
    for item in config["new_keys"]:
        x=generate_pn_sequence(item["key"],length).astype(float); runs=[]; start=0
        for i in range(1,len(x)+1):
            if i==len(x) or x[i]!=x[start]: runs.append(i-start); start=i
        row={"pn_key_id":item["key_id"],"effective_seed_32":item["effective_seed_32"],"chips":len(x),"fraction_positive":np.mean(x>0),"mean_run_length":np.mean(runs),"max_run_length":max(runs)}
        for lag in (1,2,4,8): row[f"autocorrelation_lag{lag}"]=np.corrcoef(x[:-lag],x[lag:])[0,1]
        rows.append(row)
    return pd.DataFrame(rows)


def physical_correlations(physical: pd.DataFrame,reliability: pd.DataFrame,pn_diag: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    cols=["active_carrier_fraction","dropped_for_chronology","dropped_for_hit_window","new_unmatched_events","changed_match_status_total","sum_squared_shift_ms2","rms_applied_shift_ms"]
    summary=physical.groupby("pn_key_id")[cols].mean().reset_index().merge(pn_diag,on="pn_key_id").merge(reliability[["pn_key_id","weighted_raw_ber"]],on="pn_key_id")
    predictors=cols+["fraction_positive","mean_run_length","max_run_length","autocorrelation_lag1","autocorrelation_lag2","autocorrelation_lag4","autocorrelation_lag8"]
    correlations=pd.DataFrame([{"predictor":x,"spearman_with_weighted_raw_ber":summary[x].rank().corr(summary.weighted_raw_ber.rank()),"keys":len(summary)} for x in predictors])
    return summary,correlations


def cv_auc(frame: pd.DataFrame,names: tuple[str,...]) -> float:
    scores=np.zeros(len(frame)); groups=frame.replay_file.astype(str).to_numpy()
    for train,test in GroupKFold(5).split(frame,frame.label,groups):
        model=rf(); model.fit(frame.iloc[train][list(names)],frame.iloc[train].label); scores[test]=model.predict_proba(frame.iloc[test][list(names)])[:,1]
    return roc_auc_score(frame.label,scores)


def detectors(features: pd.DataFrame,panel: pd.DataFrame,subset: dict) -> pd.DataFrame:
    p1f=pd.read_csv(PHASE1/"timing_features.csv"); hist=p1f[(p1f.historical_pn==1)&(p1f.layout_seed<3)]
    test=pd.concat([hist,features[features.pn_key_id.isin(subset["phase2_key_ids"])]],ignore_index=True,sort=False)
    old=pd.read_csv(RESULTS_DIR/"adaptive_layout_strong_v1"/"features.csv"); pilot=set(panel.loc[panel.eligible_7_5==1,"replay_file"].astype(str))
    old=old[(old.partition=="development")&(old.alpha_method=="sender_local_adaptive")&(old.layout=="distributed")&~old.replay_file.astype(str).isin(pilot)]
    rows=[]
    for (pn,seed),group in test.groupby(["pn_key_id","layout_seed"]):
        train=old[old.layout_seed.astype(str)==str(seed)]
        for label,names in FEATURE_SETS.items():
            model=rf(); model.fit(train[list(names)],train.label); score=model.predict_proba(group[list(names)])[:,1]
            rows.append({"pn_key_id":pn,"layout_seed":seed,"feature_set":label,"attacker":"historical_pn_trained_replay_disjoint","roc_auc":roc_auc_score(group.label,score),"replays":group.replay_file.nunique()})
            rows.append({"pn_key_id":pn,"layout_seed":seed,"feature_set":label,"attacker":"key_aware_replay_grouped_cv","roc_auc":cv_auc(group.reset_index(drop=True),names),"replays":group.replay_file.nunique()})
    return pd.DataFrame(rows)


def collision_analysis() -> pd.DataFrame:
    space=2**32
    return pd.DataFrame([{"keys":n,"seed_space":space,"approx_collision_probability":1-math.exp(-n*(n-1)/(2*space))} for n in (40,100,1000,10000)])


def main() -> None:
    p=pd.read_csv(OUTPUT/"physical_results.csv"); m=pd.read_csv(OUTPUT/"message_results.csv"); b=pd.read_csv(OUTPUT/"bit_results.csv"); w=pd.read_csv(OUTPUT/"codeword_results.csv"); f=pd.read_csv(OUTPUT/"timing_features.csv")
    panel=pd.read_csv(OUTPUT/"replay_panel.csv"); config=json.loads((OUTPUT/"phase2_pn_keys.json").read_text()); subset=json.loads((OUTPUT/"detector_subset_lock.json").read_text())
    rel=summarize_keys(p,m); rel.to_csv(OUTPUT/"reliability_by_pn.csv",index=False); distribution(rel).to_csv(OUTPUT/"reliability_distribution.csv",index=False)
    thresholds={"ber_gt_4_5":rel.weighted_raw_ber>.045,"ber_gt_5_0":rel.weighted_raw_ber>.05,"ber_gt_5_5":rel.weighted_raw_ber>.055,"ber_gt_6_0":rel.weighted_raw_ber>.06,
        "correct_accept_lt_80":rel.correct_accept_rate<.8,"reject_gt_20":rel.reject_rate>.2,"reject_gt_25":rel.reject_rate>.25}
    pd.DataFrame([{"criterion":k,"keys_meeting":int(v.sum()),"keys_tested":len(rel)} for k,v in thresholds.items()]).to_csv(OUTPUT/"reliability_tail_counts.csv",index=False)
    combined,combined_messages,combined_summary=common_design(p,m); combined_summary.to_csv(OUTPUT/"combined_pn_distribution.csv",index=False)
    combined[["phase","config_id","replay_file","pn_key_id","layout_seed","raw_bit_errors","coded_bits"]].to_csv(OUTPUT/"phase1_phase2_common_design.csv",index=False)
    bootstrap_mean(p,m).to_csv(OUTPUT/"two_level_bootstrap.csv",index=False); replay_uncertainty(p).to_csv(OUTPUT/"replay_uncertainty_by_pn.csv",index=False)
    lm,lc=layout_analysis(p); lm.to_csv(OUTPUT/"pn_layout_matrix.csv",index=False); lc.to_csv(OUTPUT/"pn_layout_correlations.csv",index=False)
    rel[["pn_key_id","correct_accept","wrong_accept","reject","correct_accept_rate","wrong_accept_rate","reject_rate","post_ecc_ber"]].to_csv(OUTPUT/"integrity_by_pn.csv",index=False)
    conf=confidence(b,w); conf.to_csv(OUTPUT/"confidence_by_pn.csv",index=False); pn=pn_diagnostics(config,p); pn.to_csv(OUTPUT/"pn_sequence_diagnostics.csv",index=False)
    ps,pc=physical_correlations(p,rel,pn); ps.to_csv(OUTPUT/"physical_by_pn.csv",index=False); pc.to_csv(OUTPUT/"pn_metric_correlations.csv",index=False)
    detectors(f,panel,subset).to_csv(OUTPUT/"detector_by_pn.csv",index=False)
    wrong=pd.read_csv(OUTPUT/"wrong_key_sanity.csv").groupby("pn_key_id").agg(configs=("config_id","size"),wrong_key_ber=("wrong_key_ber","mean"),mean_abs_correlation=("wrong_key_mean_abs_correlation","mean")).reset_index(); wrong.to_csv(OUTPUT/"wrong_key_sanity_summary.csv",index=False)
    collision_analysis().to_csv(OUTPUT/"seed_collision_analysis.csv",index=False)
    files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(OUTPUT.iterdir()) if x.is_file() and x.name!="result_hashes.json"}
    (OUTPUT/"result_hashes.json").write_text(json.dumps({"hash_algorithm":"sha256","files":files},indent=2,sort_keys=True)+"\n")
    print("Phase2 analysis complete")


if __name__=="__main__": main()
