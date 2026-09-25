# ML Challenge 2026: Business Entity Resolution Solution

## 1. Executive summary

We indexed only the supplied Source 2 and Source 3 records, retrieved candidates from both name and address text, and used a LightGBM pair classifier with a threshold selected for entity-level macro F0.5. A correct empty prediction is explicitly rewarded in validation. No external identity data or pretrained model was used.

## 2. Methodology

### Problem analysis

Training has 2,206,821 Source 1 entities and 10,320,219 target records. Test has 1,732,544 Source 1 entities and 9,969,589 targets. In training, 5.59% of Source 1 entities are singletons and 89.01% have multiple matches. On a 69,439-link sample, only 25.70% of names and 8.17% of addresses match exactly after basic punctuation normalization. Some positive pairs have unrelated or translated names but nearly identical addresses. France occurs in test only, so the model uses Unicode-aware similarities and open-set country equality.

### Solution strategy

Disk-backed FTS5 retrieval provides independent name and address candidates. A fuzzy rank cap leaves 75 candidates per Source 1 entity. A LightGBM model scores all final candidates independently; all scores above a validated threshold are output. No one-match constraint is imposed.

## 3. Candidate generation

The target TSVs are read with `sep="\t"` into SQLite FTS5 indexes. Document frequencies identify rare name and address tokens. Queries use rare token singles, pairs, and address number-plus-token intersections, with an arbitrary country string as a filter. Name and address results are unioned and ranked by fuzzy name or address similarity; the top 75 are passed to the classifier and written to `candidate_pairs.tsv`.

On a 2,167-entity sample, name retrieval alone found 58.17% of true links, address retrieval alone 79.67%, and their union 91.13%. The top-75 cap retained 90.96%; on the first model holdout, candidate recall was 90.20%. The remaining retrieval misses are the principal recall ceiling.

## 4. Matching model

There are 36 numeric pair features covering normalized edit, token-set, token-sort, partial, Jaro-Winkler, legal-suffix, token overlap, address numeric agreement and conflict, missingness, and name-address interactions. IDs are never features. The chosen model is LightGBM with 350 trees, learning rate 0.04, 63 leaves, minimum child samples 100, column fraction 0.85, row fraction 0.85, and L2 regularization 2. LightGBM is MIT licensed and has no pretrained parameters.

Training and validation entities are disjoint, selected deterministically from Source 1 IDs. The final threshold 0.537 maximized exact entity-level macro F0.5 on the first holdout. The second disjoint holdout used the fixed threshold. A Source 1 entity with no candidate above threshold receives an empty list. Multiple candidates above threshold are all retained.

## 5. Results and error analysis

First holdout: macro F0.5 **0.894232**, precision **0.967516**, recall **0.810441**, singleton accuracy **0.878788**. Second holdout with fixed threshold: macro F0.5 **0.901287**, precision **0.965491**, recall **0.819216**, singleton accuracy **0.913386**. In the first holdout, 753 missed true links were absent from the final candidate set and 703 were retrieved but scored below threshold. There were 209 false positive links; 30 had conflicting numeric address tokens and 14 had missing target addresses. Remaining difficult cases include transliterated names and sparse or heavily reordered addresses.

Weighted similarity (0.6580) and logistic regression (0.8014) trailed LightGBM. The 50-candidate LightGBM scored 0.8836; the 75-candidate version scored 0.8913. A deeper LightGBM reached 0.8942 and led on the second holdout. Removing address, name, or numeric address features reduced score; hard-negative-only 5:1 sampling performed especially poorly. See `reports/experiment_results.csv` for all executed experiments.

Extra minimum-top-score and top-versus-second-score margin rules for one-link predictions were tested. They did not improve macro F0.5 over the selected global threshold, so no extra singleton rule was applied.

## 6. Reproduction and limitations

Run `python run_pipeline.py all --data-root <path-to-student_resource/dataset>` from `code/business_entity_resolution/` after installing `requirements.txt`. The pipeline writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and invokes the supplied validator. Its disk-backed indexes allow the large dataset to run on limited RAM, though full inference is time consuming. The final fit used a deterministic sample of 13,254 labeled Source 1 entities rather than all 2.2 million. France has no labels for validation, so unseen-country accuracy remains uncertain.
