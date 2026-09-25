# Selected model and validation

Selected LightGBM: {'n_estimators': 350, 'learning_rate': 0.04, 'num_leaves': 63, 'min_child_samples': 100, 'colsample_bytree': 0.85, 'subsample': 0.85, 'reg_lambda': 2, 'verbosity': -1, 'random_state': 42, 'n_jobs': 4}. First entity holdout: macro F0.5 0.894232, precision 0.967516, recall 0.810441, singleton accuracy 0.878788, candidate recall 0.901966, threshold 0.537. Second disjoint holdout with that fixed threshold: macro F0.5 0.901287, precision 0.965491, recall 0.819216, singleton accuracy 0.913386, candidate recall 0.902484.

The final artifact was fit on the selected labeled training sample after validation. The sample contains 13,254 Source 1 entities and 985,558 candidate pairs. The competition's full 2.2 million training entities were not all used for model fitting; this is a scalability limitation.

## Candidate generation

SQLite FTS5 Unicode token retrieval from names and addresses, using rare token and token-pair queries. Country is treated as an arbitrary string. Raw blocker recall on a 2,167-entity sample: 91.13% at K=50; name-only 58.17%; address-only 79.67%. A 75-candidate final cap retains 90.96% on that sample, and 90.20% on the first model holdout.

At K=50, the raw union averaged 180.8 candidates per Source 1 entity (median 184, maximum 339). The final model input averaged 74.36 (median 75, maximum 75), a 99.99928% reduction from all 10,320,219 training targets. The sample blocking run took 84.6 seconds for 2,167 Source 1 entities.

## Error analysis on first holdout

False positives: {"total": 209, "name_ratio_above_0_9": 58, "address_ratio_above_0_9": 51, "numeric_conflict": 30, "candidate_missing_address": 14}. False negatives: {"blocking_miss": 753, "candidate_scored_below_threshold": 703, "low_name_ratio": 245, "low_address_ratio": 400, "candidate_missing_address": 252}. Missing candidates account for 753 of all missed links; the rest were retrieved but scored below the selected threshold. The main remaining retrieval risk is a target with both a translated or changed name and heavily reformatted or sparse address. Numeric address agreement matters: removing numeric address features reduced macro F0.5 to 0.8658.

Most used features by LightGBM split importance: [('name_partial', 1737), ('name_len_ratio', 1660), ('name_legal_ratio', 1402), ('name_jaro', 1359), ('name_jaccard', 1283), ('address_token_set', 1227), ('address_len_ratio', 1190), ('address_partial', 1129), ('name_token_sort', 1085), ('name_token_set', 1066), ('address_token_sort', 1037), ('address_jaccard', 976)].

## Test inference totals

All 1,732,544 test entities have 75 or fewer candidates (128,805,516 candidate IDs total; mean 74.34; no empty candidate lists). The final file contains 5,394,722 links and 123,098 empty predictions. Empty prediction counts by country: US 44,127 of 663,106; France 11,841 of 259,452; India 67,130 of 809,986. These are predictions, not test accuracy estimates because test labels are unavailable.

## Experiments

Weighted similarity and logistic regression scored 0.6580 and 0.8014 at the 75-candidate cap. LightGBM small scored 0.8913; deeper scored 0.8942 and 0.9013 on the two holdouts. Shallower trees, 5:1 hard-negative-only sampling, and removal of name, address, or numeric features worsened validation. A 10:1 mixed hard-negative sample scored 0.8904. The 75-candidate cap improved score over the 50-candidate cap (0.8836) with a moderate inference-cost increase. All candidate negatives were used for the selected model.

Singleton-specific minimum-top-score and top-versus-second-score margin rules were also tested. Neither improved the first holdout macro F0.5 over the global threshold; see `singleton_rules.csv`.
