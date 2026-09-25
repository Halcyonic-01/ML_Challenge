# Business entity resolution solution

This project matches each test Source 1 business to zero or more records in test Sources 2 and 3. It uses only the competition TSV files. The matching model is LightGBM 4.7.0 (MIT license, no pretrained parameters).

## Setup

Use Python 3.12 on Windows and install the pinned packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The supplied dataset stays in its original location. `--data-root` must point to its `dataset` folder, which contains `train/` and `test/`; the supplied validator must be in the sibling `utils/` folder.

## Regenerate everything

From this project folder:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py all --data-root "C:\Users\YASH\Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset"
```

The command builds disk-backed FTS indexes in `cache/`, reads the training TSVs, runs entity-level validation and model selection, retrains the selected model on the sampled labeled entities, predicts every test Source 1 entity, and runs the supplied validator. It may require several hours and several gigabytes of free disk because the test set has nearly 10 million candidate records and 1.73 million Source 1 records. Intermediate indexes and shards can be deleted after a successful run; `models/`, `reports/`, and `output/` are the deliverables.

When running from the packaged folder `code/business_entity_resolution/`, add `--output-dir "..\..\output"` to write regenerated files into the submission zip's top-level `output/` folder.

To run stages separately:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py train --data-root "C:\Users\YASH\Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset"
.\.venv\Scripts\python.exe run_pipeline.py predict --data-root "C:\Users\YASH\Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset"
.\.venv\Scripts\python.exe run_pipeline.py validate --data-root "C:\Users\YASH\Desktop\6ab10eb3b23ba_student_resource\student_resource\dataset"
```

## Method

`src/build_index.py` loads target TSVs with `pd.read_csv(..., sep="\t")` and builds one SQLite FTS5 index per split. `src/blocking.py` issues Unicode-aware, country-filtered name and address token queries. It chooses rare tokens using corpus document frequency. A cheap fuzzy rank caps the union at 75 candidates per Source 1 entity. The cap is the exact candidate set sent to the classifier and written to `candidate_pairs.tsv`.

`src/features.py` creates 36 country-agnostic pair features: separate Unicode-normalized name and address similarities, legal-suffix normalization, token agreement, numeric address agreement/conflict, missingness, retrieval channel indicators, and interactions. IDs identify rows only and never enter model features. Country is treated as an open-set string; France is never filtered out.

`src/train_model.py` selects a deterministic labeled sample of training Source 1 entities, then uses disjoint Source 1 entities for train and validation. All candidate negatives are retained, including highly similar false matches. Baseline weighted similarity, logistic regression, and several LightGBM variants are compared. `src/evaluate.py` searches thresholds on **exact per-entity macro F0.5**, scoring correct empty predictions as 1.0. The selected model is refit on the full labeled sample after validation. The final rule accepts every candidate with model probability at least 0.537; it allows multiple matches and naturally emits an empty list for singletons.

The selected first holdout score is macro F0.5 **0.894232**, precision **0.967516**, recall **0.810441**, candidate recall **0.901966**, and singleton accuracy **0.878788**. On a second disjoint holdout, the same threshold gave macro F0.5 **0.901287**. See `reports/` for the experiment table, blocking and candidate-cap study, threshold curve, data profile, and error analysis.

The final model was fit on 13,254 sampled Source 1 training entities, not all 2.2 million. This kept feature generation and validation tractable on a 16 GB machine; the fitted sample contains roughly 986,000 labeled candidate pairs. That sample-size choice and France's absence from training are the principal generalization risks.

## Outputs and validation

`output/candidate_pairs.tsv` has one row for every test Source 1 entity and columns `source1_entity_id` and `candidate_entity_ids`. `output/matching_results.tsv` has the same Source 1 coverage and columns `source1_entity_id` and `matched_entity_ids`. Empty lists are blank cells; IDs inside lists are comma-separated. Both files are TSV.

`run_pipeline.py validate` checks both files with a memory-bounded streaming validator, including full Source 1 coverage and match-subset-candidate. It also runs the supplied validator on the matching file. The supplied validator loads all candidate IDs into RAM when given `--candidate`; with 1.73 million rows and up to 75 candidates each, that option may exceed a 16 GB machine. On a larger-memory machine, its direct command from the supplied `student_resource` folder is:

```powershell
python utils\validate_submission.py --matching "<project>\output\matching_results.tsv" --candidate "<project>\output\candidate_pairs.tsv" --test-dir dataset\test
```

The complete experiment details, including comparisons that performed worse, are in `reports/experiment_results.csv`; no external lookup or pretrained embedding model was used.
