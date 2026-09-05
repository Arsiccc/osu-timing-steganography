| claim_id | metric | baseline | final_or_ecc | cost | system_spec_sha256 | source_artifact |
| --- | --- | --- | --- | --- | --- | --- |
| ECC_001 | equal-useful full recovery gain | 72.0% uncoded | 90.0% SECDED | 50% code rate | b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038 | results/ecc_equal_payload_v1/paired_recovery.csv |
| INT_001 | wrong accept | 14/835 | 1/835 | rejection 54→118 | b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038 | results/ecc_integrity_validation_v1/summary.csv |
| INT_002 | correct accept | 767/835 | 716/835 | 51 additional correct rejects | b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038 | results/ecc_integrity_validation_v1/silent_error_capture.csv |
