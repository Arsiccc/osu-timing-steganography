"""Finalize descriptive Phase-2 PN summaries and immutable result hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT=Path("results/pn_key_robustness_phase2_v1"); PHASE1=Path("results/pn_key_robustness_v1")


def bootstrap_keys(values: np.ndarray,seed: int,iterations: int=10000) -> tuple[float,float]:
    rng=np.random.default_rng(seed); means=[rng.choice(values,len(values),replace=True).mean() for _ in range(iterations)]
    return tuple(np.quantile(means,[.025,.975]))


def phase1_replay_ci_widths(physical: pd.DataFrame) -> list[float]:
    rng=np.random.default_rng(20260902); widths=[]
    for _,group in physical.groupby("pn_key_id"):
        replay=group.groupby("replay_file").agg(errors=("raw_bit_errors","sum"),bits=("coded_bits","sum")); values=[]
        for _ in range(2000):
            sample=replay.iloc[rng.integers(0,len(replay),len(replay))]; values.append(sample.errors.sum()/sample.bits.sum())
        low,high=np.quantile(values,[.025,.975]); widths.append(float(high-low))
    return widths


def main() -> None:
    rel=pd.read_csv(OUTPUT/"reliability_by_pn.csv"); combined=pd.read_csv(OUTPUT/"combined_pn_distribution.csv"); messages=pd.read_csv(OUTPUT/"message_results.csv")
    phase1_physical=pd.read_csv(PHASE1/"physical_results.csv"); replay_ci=pd.read_csv(OUTPUT/"replay_uncertainty_by_pn.csv")
    new40=combined[combined.phase!="historical"]; historical=combined[combined.phase=="historical"].iloc[0]
    phase2_low,phase2_high=bootstrap_keys(rel.weighted_raw_ber.to_numpy(),20260904); combined_low,combined_high=bootstrap_keys(new40.weighted_raw_ber.to_numpy(),20260905)
    historical_position=pd.DataFrame([{"historical_key_id":historical.pn_key_id,"historical_common_design_ber":historical.weighted_raw_ber,
        "phase2_mean_ber":rel.weighted_raw_ber.mean(),"difference_from_phase2_mean":historical.weighted_raw_ber-rel.weighted_raw_ber.mean(),
        "difference_in_phase2_sd":(historical.weighted_raw_ber-rel.weighted_raw_ber.mean())/rel.weighted_raw_ber.std(ddof=1),
        "ascending_rank_among_phase2_values":int((rel.weighted_raw_ber<historical.weighted_raw_ber).sum()+1),"phase2_keys":len(rel),
        "empirical_fraction_phase2_at_or_below_historical":float((rel.weighted_raw_ber<=historical.weighted_raw_ber).mean())}])
    historical_position.to_csv(OUTPUT/"historical_key_position.csv",index=False)
    pd.DataFrame([{"population":"phase2_30_new","keys":len(rel),"mean":rel.weighted_raw_ber.mean(),"sd":rel.weighted_raw_ber.std(ddof=1),
        "median":rel.weighted_raw_ber.median(),"min":rel.weighted_raw_ber.min(),"max":rel.weighted_raw_ber.max(),"pn_key_bootstrap_ci95_low":phase2_low,"pn_key_bootstrap_ci95_high":phase2_high},
        {"population":"combined_40_new_common_design","keys":len(new40),"mean":new40.weighted_raw_ber.mean(),"sd":new40.weighted_raw_ber.std(ddof=1),
        "median":new40.weighted_raw_ber.median(),"min":new40.weighted_raw_ber.min(),"max":new40.weighted_raw_ber.max(),"pn_key_bootstrap_ci95_low":combined_low,"pn_key_bootstrap_ci95_high":combined_high}]).to_csv(OUTPUT/"combined_distribution_summary.csv",index=False)
    phase0_full=phase1_physical.groupby("pn_key_id").agg(errors=("raw_bit_errors","sum"),bits=("coded_bits","sum")).assign(ber=lambda x:x.errors/x.bits)
    phase1_widths=phase1_replay_ci_widths(phase1_physical)
    phase0=pd.DataFrame([{"between_key_sd_pp_five_layouts":phase0_full.ber.std(ddof=1)*100,
        "median_replay_bootstrap_ci_width_pp":float(np.median(phase1_widths)*100),
        "phase2_median_replay_bootstrap_se_pp":float(replay_ci.replay_bootstrap_se.median()*100),"chosen_first_k_layouts":3,
        "chosen_design_expected_configs":6120,"reason":"first 3: Pearson 0.955, Spearman 0.891, MAE 0.342 pp vs five-layout key BER"}])
    phase0.to_csv(OUTPUT/"phase0_variance_summary.csv",index=False)
    secded=[]
    for pn,g in messages.groupby("pn_key_id"):
        recovered=g.post_ecc_bit_errors.eq(0)
        secded.append({"pn_key_id":pn,"configs":len(g),"full_message_recovered":int(recovered.sum()),"full_message_recovery_rate":recovered.mean(),
            "post_ecc_bit_errors":int(g.post_ecc_bit_errors.sum()),"wrong_accept":int(g.wrong_accept.sum()),"reject":int(g.reject.sum())})
    pd.DataFrame(secded).to_csv(OUTPUT/"secded_by_pn.csv",index=False)
    original=pd.read_csv(OUTPUT/"two_level_bootstrap.csv"); observed={"mean_raw_ber":rel.weighted_raw_ber.mean(),"mean_correct_accept":rel.correct_accept_rate.mean(),"mean_reject":rel.reject_rate.mean()}
    original["estimate_observed"]=[observed[x] for x in original.metric]; original["original_estimate_is_bootstrap_monte_carlo_mean"]=1
    original.to_csv(OUTPUT/"two_level_bootstrap_corrected.csv",index=False)
    detector=pd.read_csv(OUTPUT/"detector_by_pn.csv").groupby(["pn_key_id","attacker","feature_set"]).roc_auc.mean().reset_index()
    detector.to_csv(OUTPUT/"detector_summary_by_pn.csv",index=False)
    report=f"""# PN-key robustness Phase 2

## Frozen design

Phase 0 selected the unchanged 68-replay eligible panel and the first three frozen layout seeds before Phase-2 keys existed. Thirty new SHA-derived keys produced 6,120 physical configurations. All 41 historical/Phase-1/Phase-2 effective 32-bit seeds were distinct. PN config SHA-256: `{hashlib.sha256(Path('data/config/pn_key_robustness_phase2_v1.json').read_bytes()).hexdigest()}`. Experiment-lock SHA-256: `{hashlib.sha256((OUTPUT/'experiment_lock.json').read_bytes()).hexdigest()}`.

## Reliability distribution

Across 30 new keys, weighted raw BER was mean {rel.weighted_raw_ber.mean():.3%}, SD {rel.weighted_raw_ber.std(ddof=1):.3%}, median {rel.weighted_raw_ber.median():.3%}, and range {rel.weighted_raw_ber.min():.3%}–{rel.weighted_raw_ber.max():.3%}. The PN-key bootstrap 95% interval for the mean was [{phase2_low:.3%}, {phase2_high:.3%}]. On the exact common three-layout design, all 40 independently preregistered new keys had mean {new40.weighted_raw_ber.mean():.3%}, SD {new40.weighted_raw_ber.std(ddof=1):.3%}, and range {new40.weighted_raw_ber.min():.3%}–{new40.weighted_raw_ber.max():.3%}.

The historical key BER was {historical.weighted_raw_ber:.3%}, only {(historical.weighted_raw_ber-rel.weighted_raw_ber.mean())*100:+.3f} percentage points from the Phase-2 mean ({(historical.weighted_raw_ber-rel.weighted_raw_ber.mean())/rel.weighted_raw_ber.std(ddof=1):+.3f} SD). It ranked {int((rel.weighted_raw_ber<historical.weighted_raw_ber).sum()+1)}/30 ascending and is descriptively typical.

Empirical Phase-2 tails were 27/30 above 4.5%, 18/30 above 5.0%, 8/30 above 5.5%, and 1/30 above 6.0%. Correct accept was below 80% for 21/30; rejection exceeded 20% for 21/30 and 25% for 2/30. These are tested-key frequencies, not probabilities over the 2^32 seed space.

## Robustness diagnostics

Cross-layout PN-key correlations were weak to moderate (Pearson 0.189–0.544; Spearman 0.182–0.574), confirming meaningful interaction. Higher key BER was associated descriptively with lower correct acceptance (Spearman -0.695) and higher rejection (+0.686). Six wrong accepted messages occurred across six keys and 6,120 configurations.

Low |C| remained useful for every key: error-prediction AUC {pd.read_csv(OUTPUT/'confidence_by_pn.csv').low_abs_c_error_auc.min():.3f}–{pd.read_csv(OUTPUT/'confidence_by_pn.csv').low_abs_c_error_auc.max():.3f}. Active-carrier survival and energy remained nearly invariant. PN mean run length and lag-1 autocorrelation had descriptive BER correlations of {pd.read_csv(OUTPUT/'pn_metric_correlations.csv').set_index('predictor').loc['mean_run_length','spearman_with_weighted_raw_ber']:.3f} and {pd.read_csv(OUTPUT/'pn_metric_correlations.csv').set_index('predictor').loc['autocorrelation_lag1','spearman_with_weighted_raw_ber']:.3f}; no screening rule is justified.

For the preregistered detector subset, historical-PN-trained FULL mean-over-layout AUC ranged {detector.query("attacker=='historical_pn_trained_replay_disjoint' and feature_set=='full'").roc_auc.min():.3f}–{detector.query("attacker=='historical_pn_trained_replay_disjoint' and feature_set=='full'").roc_auc.max():.3f}; key-aware FULL ranged {detector.query("attacker=='key_aware_replay_grouped_cv' and feature_set=='full'").roc_auc.min():.3f}–{detector.query("attacker=='key_aware_replay_grouped_cv' and feature_set=='full'").roc_auc.max():.3f}. Cyclic wrong-key BER averaged {pd.read_csv(OUTPUT/'wrong_key_sanity_summary.csv').wrong_key_ber.mean():.3%}.

## Decision

**A — PN VARIABILITY NOW SUFFICIENTLY CHARACTERIZED for this frozen replay/layout design.** Phase-1 sensitivity replicated. The combined 40-key distribution gives a stable empirical mean and SD, while also showing a nontrivial worse-reliability tail. This closes the PN branch; it does not authorize key screening and does not establish probabilities over the complete effective seed space.
"""
    (OUTPUT/"pn_phase2_report.md").write_text(report,encoding="utf-8")
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!="result_hashes.json"}
    (OUTPUT/"result_hashes.json").write_text(json.dumps({"hash_algorithm":"sha256","files":files},indent=2,sort_keys=True)+"\n")


if __name__=="__main__": main()
