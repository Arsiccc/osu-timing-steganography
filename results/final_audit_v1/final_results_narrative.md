# Final results narrative

## Dataset and timing reconstruction

The active corpus contains 949 replay files on 34 maps after quarantining 51 files on three maps that were incompatible with the single-offset model; quarantine is not evidence of corruption. [CLAIM:DATA_001]

## Residual matching quality

Raw cumulative replay timing, one fixed offset per map, and optimal monotone matching yielded 99.22% weighted matching on the active corpus, with a 100% median replay ratio. [CLAIM:MATCH_001]

## Baseline embedding reliability

The five-layout adaptive+DISTRIBUTED physical channel had 1,986 errors among 47,130 coded bits (4.21% BER). [CLAIM:DECODER_001]

## Adaptive alpha

Sender-local alpha uses replay-header hit counts and exact rational boundaries; the corrected v2 semantics changed no assignment in the 949-replay corpus. The frozen validation point estimate was lower than its fixed-alpha control under FULL-29. [CLAIM:ADAPT_001]

## Strong steganalysis and PREFIX placement weakness

FULL-29 exposed a strong PREFIX-specific position signature. [CLAIM:DIST_001]

## DISTRIBUTED placement

Across five frozen layouts, DISTRIBUTED substantially reduced PREFIX AUC, while unseen-map FULL-29 AUC remained 0.597–0.614 rather than chance. [CLAIM:DIST_001] [CLAIM:DIST_002]

## Decoder error structure

Errors were mildly bursty rather than demonstrably independent. Low absolute correlation was a useful error indicator (AUC 0.854), but not a calibrated probability. [CLAIM:DECODER_001] [CLAIM:CONF_001]

## SECDED

At equal useful payload, SECDED improved full-message recovery by 18.0 percentage points (95% replay-cluster CI 12.75–23.25 pp), at 50% coding rate and greater physical cost. [CLAIM:ECC_001]

## Confidence-based integrity rejection

On 835 unseen-map validation configurations, the observable integrity rule reduced wrong accepts from 14 to 1 while correct accepts fell from 767 to 716 and rejects rose from 54 to 118. [CLAIM:INT_001] [CLAIM:INT_002]

## Capacity / abstention

The new-corpus normal class achieved 2.55% raw BER and 91.41% correct acceptance; the newly covered floor class had 5.89% BER and 78.67% correct acceptance, so the floor was rejected. [CLAIM:FLOOR_001]

## PN robustness

Across 30 new PN keys, mean BER was 5.136%, between-key SD 0.478 pp, and range 4.087–6.190%; the historical PN was descriptively typical. [CLAIM:PN_001] [CLAIM:PN_002]

## Message-content robustness

Five messages produced BER 5.769–7.131% with 0.498 pp between-message SD, supporting modest rather than catastrophic content sensitivity. [CLAIM:MSG_001]

## Classifier-family robustness

FULL-29 AUC remained 0.603–0.635 across RF, Logistic Regression, RBF SVM, and HGB in replay-grouped and unseen-map regimes. [CLAIM:CLF_001]

## Final system

The frozen default is adaptive-alpha v2 + DISTRIBUTED + N=8 + normal 7.5% coded-bit allocation + SECDED(8,4) + observable integrity rejection + abstention when no complete word fits. It has modest, above-chance detectability under tested adversaries and is not a cryptographic protocol.

## Limitations

Evidence is corpus- and detector-bounded; robustness panels reuse consumed data; short-file abstention limits coverage; and PN/message variation remains measurable.
