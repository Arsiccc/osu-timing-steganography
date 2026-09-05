# Abstract-level claims

1. A validated raw-delta, fixed-map-offset reconstruction achieved 99.22% weighted matching on 949 active osu!standard replay files. [CLAIM:MATCH_001]
2. A 29-feature detector revealed a strong PREFIX-specific timing signature, and keyed DISTRIBUTED placement substantially reduced it across five frozen layouts. [CLAIM:DIST_001]
3. DISTRIBUTED embedding retained modest above-chance detectability on unseen maps (FULL-29 AUC 0.597–0.614), so the method is not described as undetectable. [CLAIM:DIST_002]
4. Physical channel errors were mildly bursty, while low absolute PN correlation was a useful bit-error indicator (AUC 0.854). [CLAIM:DECODER_001] [CLAIM:CONF_001]
5. SECDED(8,4) increased equal-useful-payload full-message recovery by 18.0 pp, with a 50% coding-rate cost. [CLAIM:ECC_001]
6. An observable integrity rule reduced wrong acceptance from 14/835 to 1/835 on frozen unseen-map validation, while increasing rejection. [CLAIM:INT_001]
7. A 12.5%-capped short-replay floor was rejected after materially poorer new-corpus reliability. [CLAIM:FLOOR_001]
8. Reliability varied across preregistered PN realizations and, more modestly, across message content; neither dimension justified post-hoc screening. [CLAIM:PN_001] [CLAIM:MSG_001]
9. Modest detectability persisted across four classifier families and was not specific to Random Forest under the tested timing features. [CLAIM:CLF_001]
10. Results remain bounded to tested corpora, hand-engineered timing features, and a non-cryptographic 32-bit PN seed.
