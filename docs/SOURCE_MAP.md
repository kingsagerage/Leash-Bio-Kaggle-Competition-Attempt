# Documentation source map

## Primary project source

The project description is grounded in `Pasted markdown.md`, preserved byte-for-byte as [source/architecture_excerpt.md](source/architecture_excerpt.md).

- Original source length: 289 lines.
- SHA-256: `150ab4cd4857fe09555b941045fa69cadaeab92d680c931fcd43fd441b77f758`.
- Lines below refer to the original file, counting blank lines and code fences.
- Only an architecture excerpt was supplied for this project; the older Kaggriculture files are unrelated and were not used as Leash Bio model evidence.

| README claim | Source lines | Evidence category |
| :--- | :--- | :--- |
| Python packages imported | 2-18 | Code |
| One-million-row CSV prefix; 100,000-row comment mismatch | 23-24, 168 | Code and displayed output |
| SMILES parsed with RDKit | 29-30 | Code |
| Morgan fingerprint radius 3 and 1,024 bits | 35-41 | Code |
| BRD4, HSA, sEH; repeated molecule across protein rows | 50-56 | Displayed preview |
| pandas protein indicators | 59-63, 73-76 | Code and displayed output |
| Fingerprints expanded to 1,024 columns | 92-108 | Code |
| Numeric table has 1,029 columns | 116-168 | Displayed output |
| 1,027 pre-PCA features | 92-108, 185-190 | Derived: 1,024 + 3, excluding id/label |
| 400-component PCA; no StandardScaler call | 173-201 | Code |
| Saved explained-variance value of 73.56% | 206-213 | Stored output; not rerun |
| Label and ID removed from model inputs | 218-226 | Code |
| Row-wise 80/20 outer split with seed 42 | 231-232 | Code |
| Inconsistent 200-column X_test display | 244-248 | Stored output |
| 997,424 negative and 2,576 positive rows | 249-252 | Stored output |
| Dense layers and sigmoid output | 257-271 | Code |
| Adam 0.001, BCE, accuracy, precision | 273-276 | Code |
| 30 requested epochs; batch 32; internal validation 0.2 | 278-282 | Code |
| Evaluation calls, without printed results | 284-288 | Code; results absent |

## Derived quantities

These quantities are calculations, not measurements from an executed training run:

```text
Feature width before PCA = 1,024 + 3 = 1,027
Positive fraction = 2,576 / 1,000,000 = 0.2576%
All-negative accuracy on the displayed sample = 997,424 / 1,000,000 = 99.7424%
Outer training partition = 1,000,000 * 0.8 = 800,000 rows
Keras validation partition = 800,000 * 0.2 = 160,000 rows
Optimization partition = 800,000 * 0.8 = 640,000 rows
Outer test partition = 1,000,000 * 0.2 = 200,000 rows
```

The split counts assume all one million rows survive preprocessing.

Dense parameter counts, including biases:

```text
400 -> 128: (400 + 1) * 128 = 51,328
128 -> 256: (128 + 1) * 256 = 33,024
256 -> 256: (256 + 1) * 256 = 65,792, repeated 3 times
256 -> 128: (256 + 1) * 128 = 32,896
128 ->   1: (128 + 1) *   1 =    129
Total: 314,753
```

## External context, not project evidence

| Source | What it supports |
| :--- | :--- |
| [Official Kaggle competition](https://www.kaggle.com/competitions/leash-BELKA) | Full title, host, April 4-July 8, 2024 timeline, binary submission format, and AP aggregation over `(protein, split group)`. The same page was also retrieved through Kaggle's `/c/leash-BELKA` route. |
| [RDKit getting started](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) | Morgan/circular fingerprint terminology and representation. |
| [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | Centering versus unit-variance scaling. |
| [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | Fit preprocessing only on training data; reuse transforms for held-out data. |
| [Keras model training APIs](https://keras.io/api/models/model_training_apis/) | Meaning of validation_split and its selection before shuffling. |
| [Keras classification metrics](https://keras.io/api/metrics/classification_metrics/#precision-class) | Threshold precision is distinct from ranked average precision. |
| [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) | Definition and use of AP for score-based evaluation. |

External sources were checked while drafting. Their package documentation versions are not evidence of the original experiment's installed versions.

## Graphics provenance

All diagrams were produced for this documentation from the supplied code and the arithmetic above. The abstract hero motif is not a molecular structure, a PCA plot, or a learned embedding. There are no invented learning curves, binding measurements, or leaderboard graphics. The feature and training diagrams show the supplied workflow, including PCA before splitting. Proposed changes appear only in the separately labeled validation notes.

The style source is [diagrams/render_graphics.py](diagrams/render_graphics.py), with editable Mermaid equivalents for the three process diagrams. SVGs and PNGs contain the same diagram content.
