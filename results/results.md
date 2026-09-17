# Evaluation results

Model `qwen3:4b-instruct-2507-q4_K_M`, service `http://localhost:8000`, run 2026-09-17T11:28:34+00:00.

| metric | value |
|---|---|
| documents | 6 |
| documents_failed | 0 |
| fields_scored | 42 |
| field_accuracy | 0.81 |
| recall_when_stated | 0.852 |
| hallucination_rate_when_absent | 0.267 |
| confidently_wrong | 7 |
| flagged_unverified | 3 |
| flagged_and_actually_wrong | 0 |
| flagged_but_correct | 3 |

## Per field

| field | correct | correct_null | missed | wrong | hallucinated | confidently wrong |
|---|---|---|---|---|---|---|
| contract_type | 4 | 0 | 0 | 2 | 0 | 2 |
| parties | 5 | 0 | 0 | 1 | 0 | 1 |
| agreement_date | 6 | 0 | 0 | 0 | 0 | 0 |
| effective_date | 2 | 3 | 0 | 0 | 1 | 1 |
| expiration_date | 2 | 4 | 0 | 0 | 0 | 0 |
| renewal_term | 0 | 3 | 0 | 0 | 3 | 3 |
| governing_law | 4 | 1 | 1 | 0 | 0 | 0 |

## Per document

| document | right / 7 | seconds | prompt tokens | truncated |
|---|---|---|---|---|
| distributor_baseline | 5 | 198.9 | 2535 | False |
| transportation_three_dates | 7 | 187.2 | 3080 | False |
| sponsorship_two_jurisdictions | 7 | 114.6 | 2150 | False |
| cobranding_duration_only | 6 | 206.1 | 3043 | False |
| consulting_truncated | 4 | 291.9 | 4392 | True |
| joint_filing_not_a_commercial_contract | 5 | 65.7 | 1407 | False |

## Fields that were not right

| document | field | gold | predicted | status | reason | outcome |
|---|---|---|---|---|---|---|
| distributor_baseline | effective_date | — | 2021-03-18 | found_and_verified | — | hallucinated |
| distributor_baseline | renewal_term | — | unless sooner terminated by either party upon (30) days written notice, without cause | found_and_verified | — | hallucinated |
| cobranding_duration_only | renewal_term | — | annually renew this agreement for a period of one year | found_and_verified | — | hallucinated |
| consulting_truncated | contract_type | Other | Service | found_and_verified | — | wrong |
| consulting_truncated | renewal_term | — | one (1) year or until Consultant completes the services requested | found_and_verified | — | hallucinated |
| consulting_truncated | governing_law | Florida | — | not_found | — | missed |
| joint_filing_not_a_commercial_contract | contract_type | Other | Joint Venture | found_and_verified | — | wrong |
| joint_filing_not_a_commercial_contract | parties | ABP TRUST; ADAM D. PORTNOY | ABP TRUST | found_and_verified | — | wrong |
