# Validation notes

These notes distinguish **observations in the supplied source** from **proposed follow-up work**. They do not modify the preserved experiment or claim that any follow-up has been implemented.

## Observed in the source

| Observation | Implication / uncertainty |
| :--- | :--- |
| The loading comment says 100,000 rows; the statement uses `nrows=1000000`. | The README documents the executable statement and one-million-row output, while retaining the comment discrepancy. |
| `PCA(n_components=400)` and `Dense(..., input_dim=400)` are declared, but `X_test.columns` prints 200 columns. | The declaration is clear, but the saved outputs cannot establish a consistent executed configuration. The 73.56% explained-variance output cannot safely be attributed to a clean 400-component run. |
| Protein indicators are included in `dta_for_pca`. | Both molecular features and protein identity are projected together. This is not the architecture where only fingerprints undergo PCA and protein indicators are concatenated afterward. |
| PCA is fitted before the outer train/test split. | The transform is influenced by held-out feature distributions. Any new generalization claim requires an appropriately separated evaluation. |
| Row-wise `train_test_split` has no `stratify` or grouping argument. | Split-specific label counts, molecule overlap, and building-block overlap are unknown. |
| `generate_ecfp` can return `None`; `ndf['ecfp'].apply(len)` follows later. | Invalid SMILES are not fully handled in the provided flow. The displayed run does not establish robust behavior for new invalid records. |
| RDKit molecule objects, fingerprint lists, expanded numeric columns, and PCA data coexist. | Several large intermediate representations are retained. No exact runtime or peak-memory figure is provided. |
| `Dropout` and `EarlyStopping` are imported but not used. | Do not describe the architecture as dropout-regularized or early-stopped. |
| `StandardScaler` and `OneHotEncoder` are imported. | Standard scaling is not called, and the actual protein encoder is pandas `get_dummies`. |
| `RandomForestClassifier` and DuckDB are imported but not used in the shown fitting/loading path. | The documented model is not a random-forest ensemble, and its CSV input is not processed with DuckDB. |
| `average_precision_score` is imported but not invoked. | No local AP or official competition-score result is present. |
| `model.fit` and `model.evaluate` are shown without their resulting logs or numeric values. | The source demonstrates configured operations, not independently verified completion or predictive quality. |

Source: [architecture excerpt](source/architecture_excerpt.md). Exact locations are listed in [SOURCE_MAP.md](SOURCE_MAP.md).

## Suggested next evaluation - not the original experiment

**Start with a consistent run.** Choose the PCA width deliberately, match the model input to it, and clear stale cell outputs. Preserve a run configuration, feature schema, package versions, and all split membership indices.

**Separate data before fitted preprocessing.** Create explicit optimization, validation, and test partitions. Fit PCA only on optimization features, then use the same object to transform the other partitions. A random stratified split can be a diagnostic baseline, but it is not a substitute for measuring generalization across molecular or building-block groups. The original excerpt does not implement either improvement. See [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

**Audit chemical overlap.** Measure duplicate molecule SMILES across partitions and track shared building blocks. Consider molecule-grouped or chemistry-aware partitions as separate experiments. Document their definitions rather than calling them competition-equivalent without reproducing the organizer's split groups.

**Evaluate ranked probabilities.** Retain probability scores and report AP together with prevalence and sample counts for each evaluation group. Kaggle's metric averages AP over `(protein, split group)`, so a pooled AP or three-protein mean is not automatically the official score. Report threshold precision or recall only with the threshold specified. Sources: [competition](https://www.kaggle.com/competitions/leash-BELKA), [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

**Test modeling changes separately.** Potential ablations include a no-PCA baseline, keeping protein indicators outside PCA, different fingerprint settings, or imbalance-aware training. These are candidate experiments, not components of the supplied architecture and not promised performance improvements.

**Persist the whole prediction pipeline.** Store the ordered feature schema, protein encoding, fingerprint configuration, PCA state, model, and run metadata. Adding only neural weights would not reproduce this model's input transformation. Export and test a submission path separately from local held-out evaluation.

## What has not been verified

No training dataset, executable notebook, saved fitted PCA, neural checkpoint, epoch history, prediction file, or leaderboard result was supplied for this documentation task. This package did not train the model, reconstruct saved weights, validate a Kaggle submission, or resolve the 200/400-component inconsistency by experiment. The parameter count and split sizes in the README are arithmetic derivations from source declarations.
