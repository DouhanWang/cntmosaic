# XGBoost categorical imputation

Install from the repository root:

```sh
python -m pip install -e ".[imputation]"
```

Run imputation on raw data **before** constructing `ContactData` or
`ParticipantData`, whose preprocessing can drop records with missing values.
Merge participant predictors into contact records first, using a validated
many-to-one join. Exclude identifiers and features derived from the target.

```python
from cntmosaic.preprocess import XGBoostImputer

data = contacts.merge(
    participants, on="part_id", how="left", suffixes=("", "_part"),
    validate="m:1",
)
imputer = XGBoostImputer(
    target="cont_education",
    features=["part_age_exact", "cont_age_exact", "cont_gender", "setting"],
    feature_selection="backward",  # None skips feature selection
    cv=5,
    scoring="roc_auc_ovr",        # also accuracy or f1_macro
    class_weight=None,            # or "balanced"
    min_features=1,
    selection_tol=0.0,            # only accept non-decreasing CV scores
    random_state=42,
    model_params={"n_estimators": 100, "max_depth": 6, "n_jobs": 4},
    confidence_threshold=None,    # default: fill every missing target
    compute_validation=True,      # store out-of-fold threshold diagnostics
)
completed = imputer.fit_transform(data, groups=data["part_id"])

print(imputer.selected_features_)
print(imputer.selection_score_)  # feature-selection CV AUC, not holdout performance
print(imputer.selection_history_)
```

Omit `groups` for ordinary stratified folds. When passing groups, every fold
must contain all target classes in both training and validation partitions;
reduce `cv` if that is not feasible. Missing target records are excluded from
training and selection. The final model uses all observed targets, with no
reserved holdout. Encoders, numeric medians and balancing weights are learned
inside each training fold, then refitted on all observed targets.

`transform()` returns a copy, fills only missing targets, and preserves row
order, index and observed values. Numeric class labels retain their original
meaning. This implementation supports categorical targets (including numeric
category codes), not continuous regression or multiple imputation. Train a
separate instance for each target, with an explicit predictor list. Class
balancing changes the training objective and may change imputed proportions;
it is not survey weighting or a guarantee of calibrated probabilities.

Backward elimination can require many model fits. To use a fixed predictor
list quickly, set `feature_selection=None`. Inspect `excluded_features_` for
predictors omitted because they are entirely missing on observed-target rows.
The notebook `feature_selections.ipynb` is independent and is not changed.

## Choose a confidence threshold

Fit once, inspect coverage and reference performance, then choose a threshold:

```python
imputer.fit(data, groups=data["part_id"])

# Predictions for missing targets only. Does not change data.
preview = imputer.prediction_report(data)
print(preview)  # row_position, predicted_label, confidence (maximum probability)
print(preview["confidence"].describe())

# All class probabilities for any supplied rows; columns are original class labels.
probabilities = imputer.predict_proba(data.loc[data["cont_education"].isna()])

# Missing-record coverage: number filled, number remaining, fraction filled.
thresholds = [None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(imputer.confidence_report(data, thresholds=thresholds))

# Reference correctness and coverage from known targets' out-of-fold predictions.
validation = imputer.validation_report(thresholds=thresholds)
print(validation["summary"])
print(validation["by_class"])
print(validation["note"])

completed = imputer.transform(data, confidence_threshold=0.8)
print(imputer.imputation_summary_)  # counts from the latest transform call
```

Threshold comparison is inclusive: a probability of exactly 0.8 is accepted at
0.8. Below-threshold values remain missing. `confidence_threshold=None` fills
all missing values. Set a default in the constructor, or override per
`transform` / `fit_transform` call. Omitting the method argument uses the
constructor setting; explicitly passing None overrides it to fill everything.
Reports do not transform the data or change the last imputation summary.
Original row order and duplicate indices are preserved; `row_position` is a
zero-based position in the supplied DataFrame, not an index label.

In `confidence_report`, `coverage` is the fraction of initially missing records
that would be filled. There is no correctness estimate for those records,
because their true targets are unknown. Coverage is zero when no records are
missing. Report fractions are in [0, 1], not percentages. Threshold None may
display as NaN in a numeric pandas column; it means accept all predictions.

In `validation_report`, each known target is predicted by a fold model trained
without that row (and without its group when groups are supplied). The models
use the selected features. `summary` reports accepted/rejected counts, coverage
and accuracy among accepted predictions. `by_class` shows each true class's
support, accepted count, coverage, accuracy among accepted rows, and recall
against all rows of that true class (including abstentions as misses). It also
shows accepted predicted-class counts and precision. Undefined accuracy or
precision is NaN when no corresponding predictions are accepted.

These are **reference CV diagnostics, not independent validation** when feature
selection used the same folds. They may be optimistic, and performance on
known targets may differ from performance on missing targets. Model probability
0.8 is not a guarantee of 80% correctness. Increasing the threshold reduces
coverage; accuracy need not improve monotonically. Inspect class-level coverage
as well as overall accuracy so minority classes are not hidden. Balanced class
weights can change probabilities and the effect of a threshold. No threshold is
selected automatically.

Validation diagnostics are enabled by default and require `cv` feasible folds,
even if feature selection is disabled. They cost `cv` extra model fits after
selection. Set `compute_validation=False` to disable them; coverage reports and
probability previews still work. With both `feature_selection=None` and
`compute_validation=False`, fit performs no cross-validation.

Remaining low-confidence missing records can still be dropped by downstream
`ContactData` / `ParticipantData` preprocessing. Check `n_remaining` before
constructing those containers. Inspect `oof_predictions_` and
`oof_probabilities_` for the known-target diagnostics; these are separate from
the final model's predictions for missing records.
