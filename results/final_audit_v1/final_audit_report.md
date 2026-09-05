# Final scientific audit v1

Final status: **B — publishable with major caveats**. Frozen scientific artifacts are internally consistent and reproducible, but detectability is above chance, useful throughput is constrained by 50% ECC rate and abstention, and external-source validation is absent.

1. **Final system:** sender-local adaptive alpha v2 (exact rational thresholds), independently keyed DISTRIBUTED placement, `N=8`, normal 7.5% coded-bit allocation, SECDED(8,4), corrected-bit-not-minimum integrity rejection, and ABSTAIN when no complete word fits. Spec SHA-256: `b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`.
2. **Supporting corpora:** active 949/34 original corpus; frozen 611/23 development and 338/11 validation split; 300/20 new floor-validation primary corpus with disjoint 240-replay calibration pool; consumed 68-replay PN, 100-replay host/message, and 203-replay classifier panels.
3. **Superseded/invalid:** old timing/matching baseline, float-boundary adaptive inference, PREFIX as final layout, confidence erasure, 12.5% floor, host-aware PN selection, and any message/PN screening claim.
4. **Hashes:** all 8 available branch manifests passed (200 listed artifacts); branches without manifests are explicitly lower provenance.
5. **Semantics:** timing, matching, deduplication, writer, decoder-zero, adaptive boundaries, DISTRIBUTED, SECDED, integrity, and abstention agree. One documentation error was confirmed: 98.68% is pre-quarantine; active 949 is 99.22%.
6. **Reliability:** scoped physical BER is 4.21% on the original five-layout decoder matrix; new-corpus normal eligible cases were 2.55%. These are different corpora and are not pooled.
7. **ECC/integrity:** equal-useful SECDED recovery improved 72%→90%; integrity changed wrong accept 14→1, correct accept 767→716, and reject 54→118 of 835.
8. **Useful-payload cost:** SECDED rate is 1/2; new-corpus normal eligibility was 45%, with 4.444 useful bits/eligible configuration and 1.763 correctly accepted useful bits/input replay-layout opportunity.
9. **Detectability:** modest but above chance; adaptive DISTRIBUTED unseen-map FULL-29 AUC was 0.597–0.614 across layouts.
10. **Unseen maps:** yes, position-signature reduction and above-chance detection generalized to the frozen map-disjoint validation.
11. **Classifier families:** FULL-29 AUC was 0.603–0.635 in both replay-grouped and unseen-map regimes; not RF-specific under tested features.
12. **PN effect:** Phase-2 mean BER 5.136%, SD 0.478 pp, range 4.087–6.190%; meaningful but not catastrophic sensitivity.
13. **Message effect:** BER 5.769–7.131%, SD 0.498 pp; modest sensitivity on one-PN consumed data.
14. **Rejected methods:** PREFIX, confidence erasure, 12.5% floor, deployable host-aware PN screening, and untested message screening.
15. **Paired comparisons:** PREFIX/DISTRIBUTED, ECC controls, integrity/hard SECDED, within-panel PN, message, and classifier comparisons. The matrix marks cross-corpus comparisons NO.
16. **Non-combinable headlines:** 4.21% original physical BER, 2.55% new-corpus normal BER, 1/835 integrity validation wrong accept, and classifier robustness AUCs arise from different scopes.
17. **CI units:** replay clusters for paired/replay outcomes, beatmap clusters for unseen-map AUC, PN keys for between-PN mean, and PN+replay for Phase-2 system uncertainty.
18. **Strongest limitations:** no fully external source, above-chance detectability, hand-engineered adversaries, 32-bit non-cryptographic PN seed, abstention/throughput cost, and PN/message variation.
19. **Reviewer concerns:** strongest are selection/quarantine, development reuse, AUC≈0.6, abstention-conditioned reliability, and limited attacker family; see `reviewer_attack.md`.
20. **Figures/tables:** seven figures with metadata and six CSV/Markdown tables regenerate deterministically from frozen artifacts.
21. **Safe abstract claims:** those in `abstract_claims.md`, all claim-tagged and scoped.
22. **Forbidden claims:** undetectable, secure, authenticated, zero-risk, universally optimal, externally validated, or capacity-improved without redundancy/abstention accounting.
23. **Unresolved blockers:** no internal scientific blocker to a carefully scoped paper; external generalization and stronger adversaries remain research limitations.
24. **Project status:** **B — publishable with major caveats**.

Secret audit: PASS; environment-only API credentials. Git-history inspection was not possible because this workspace is not a Git worktree.
