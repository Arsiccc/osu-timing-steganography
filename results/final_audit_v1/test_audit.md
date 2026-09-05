# Frozen-invariant test audit

All listed checks passed under `python3 -m ...`:

- `scripts.tools.check_final_frozen_invariants`: replay/beatmap loading, raw-time plus fixed-offset relation, optimal matching sanity, note-index-preserving deduplication, writer zero-shift anchor identity, and final normal-policy abstention boundary.
- `scripts.tools.check_adaptive_alpha_v2`: all 949 assignments unchanged, exact thresholds, repeated `.osr` parsing.
- `scripts.tools.check_payload_layout`: PREFIX compatibility, DISTRIBUTED uniqueness/nesting/determinism, PN/layout independence, noiseless decoding.
- `scripts.tools.check_ecc`: exhaustive SECDED messages and one-/two-error behavior.
- `scripts.tools.check_ecc_integrity`: frozen integrity hash and decisions.
- `scripts.tools.check_decoder_zero_tie`: exact zero correlation maps to +1.
- `scripts.tools.check_one_codeword_floor_policy`: exact C=63/64/106/107 tested-variant boundaries; final system disables the floor.
- `scripts.tools.check_sender_local_adaptive`, `check_adaptive_layout_strong`, `check_ecc_equal_payload`, `check_ecc_integrity_validation`, and `check_strong_steganalysis`: experiment schemas and invariants.
- `scripts.tools.check_pn_key_robustness`, `check_pn_key_robustness_phase2`, `check_host_aware_pn_results`, `check_classifier_family_results`, and `check_message_content_final`: frozen robustness outputs.
- `scripts.analysis.finalize_research_results --check-reproducibility`: eight branch manifests and clean deterministic regeneration.

One initial direct-file invocation failed only because package imports require module execution from the repository root; the documented `python3 -m` invocation passed.
