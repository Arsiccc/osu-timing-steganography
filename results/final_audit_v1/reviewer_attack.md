# Skeptical reviewer audit

## 1. AUC around 0.6 is not stealthy.

**VALID CONCERN.** The final wording says modest above-chance detectability, never invisibility.

## 2. Development choices may overfit the original corpus.

**PARTIALLY ADDRESSED.** Map-disjoint and new-map tests exist, but sources are not fully external.

## 3. Map offsets are data-dependent.

**PARTIALLY ADDRESSED.** They are map-level and calibration-separated in the new corpus, but remain a model assumption.

## 4. Quarantining maps creates selection bias.

**VALID CONCERN.** Membership is explicit; quarantine means timing-model incompatibility, not corruption.

## 5. 7.5% is not useful payload.

**ADDRESSED BY EXPERIMENT.** It is coded-bit allocation; SECDED rate and abstention are reported separately.

## 6. SECDED throughput is costly.

**VALID CONCERN.** Useful rate is 50%, with direct physical-cost comparisons.

## 7. Abstention hides difficult short cases.

**VALID CONCERN.** Eligibility and per-input throughput are explicitly reported.

## 8. PN robustness uses one replay panel.

**VALID CONCERN.** Forty preregistered keys characterize that design only.

## 9. Message robustness uses one PN.

**VALID CONCERN.** It is labelled consumed-data robustness, not independent validation.

## 10. Classifier robustness is limited to hand-engineered features.

**VALID CONCERN.** Four families do not upper-bound learned or future attacks.

## 11. A 32-bit PN seed is not meaningful cryptographic security.

**ADDRESSED BY EXPERIMENT.** No cryptographic claim is made; it is an explicit limitation.

## 12. Observed associations may be called causal.

**ADDRESSED BY EXPERIMENT.** Narrative uses observed association and controlled comparisons only.

## 13. Integrity may reject too many correct messages.

**VALID CONCERN.** 51/767 extra false rejects and severe K=8 behavior are disclosed.

## 14. Zero new-corpus wrong accepts may be overinterpreted.

**ADDRESSED BY EXPERIMENT.** It is reported as 0/675 observed with material zero-event uncertainty.

## 15. Historical results mix incompatible corpora.

**ADDRESSED BY EXPERIMENT.** Comparability matrix and scope-labelled canonical metrics prohibit a synthetic score.
