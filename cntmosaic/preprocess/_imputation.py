"""Classification-based imputation with optional backward feature elimination."""

from __future__ import annotations

import numbers
import warnings

import numpy as np
import pandas as pd

_DEFAULT_THRESHOLD = object()


class XGBoostImputer:
    """Fill missing categorical targets using an XGBoost classifier.

    Parameters
    ----------
    target : str
        Column to impute. Numeric category labels are treated as classes, not
        continuous responses. Fit one imputer per target.
    features : sequence of str
        Explicit candidate predictors. Exclude identifiers and columns derived
        from the target. The target itself is forbidden.
    feature_selection : {'backward', None}, default='backward'
        Backward elimination evaluates every possible single-feature removal
        on fixed CV folds. Ties follow the supplied feature order.
    cv : int, default=5
        Stratified folds for selection. Supplying groups to fit uses stratified
        group folds instead (requires scikit-learn >= 1.1).
    scoring : {'roc_auc_ovr', 'accuracy', 'f1_macro'}, default='roc_auc_ovr'
        Selection score; AUC is binary AUC or multiclass macro one-vs-rest AUC.
    class_weight : {None, 'balanced'}, default=None
        Compute class-frequency weights separately in each training fold and
        on all observed targets for the final model. These are not survey weights.
    min_features : int, default=1
        Minimum predictors retained by elimination.
    selection_tol : float, default=0.0
        Maximum permitted score decrease at each removal. Zero accepts ties
        and improvements, matching non-decreasing backward elimination.
    random_state : int, default=42
        Seed for folds and models.
    model_params : dict, optional
        XGBClassifier keyword arguments, e.g. n_estimators, max_depth, n_jobs.
        Defaults to n_jobs=1. Parameters requiring an external evaluation set
        (early stopping/callbacks) are intentionally unsupported.
    confidence_threshold : float or None, default=None
        Fill only predictions whose maximum class probability is >= this
        threshold (between 0 and 1). None fills every missing target.
    compute_validation : bool, default=True
        Store out-of-fold probabilities for threshold diagnostics. This runs
        cv additional fits on the selected features, even without selection.
        False disables validation_report and skips CV when selection is None.

    Notes
    -----
    Fit uses only rows whose target is known. Numeric median imputation and
    categorical ordinal encoding are fitted within each training fold. After
    selection, preprocessing and the classifier are refitted on all known rows.
    Transform copies the input and fills only missing target values. No holdout
    is reserved. ``selection_score_`` is a feature-selection CV score, not an
    independent estimate of imputation performance. With selection disabled it
    is None. Validation diagnostics reuse selection folds and may be optimistic.
    Probabilities are uncalibrated model outputs, not guaranteed correctness.

    Attributes after fitting include selected_features_, selection_history_
    (list of removal records), excluded_features_ (all-missing predictors),
    baseline_score_, selection_score_, classes_, model_, and preprocessor_.
    XGBoost and scikit-learn are imported only when fit is called.

    Examples
    --------
    >>> imputer = XGBoostImputer(
    ...     target='cont_education',
    ...     features=['part_age', 'cont_age', 'setting'],
    ...     model_params={'n_estimators': 100},
    ... )
    >>> completed = imputer.fit_transform(data, groups=data['part_id'])
    """

    def __init__(
        self,
        target,
        features,
        feature_selection="backward",
        cv=5,
        scoring="roc_auc_ovr",
        class_weight=None,
        min_features=1,
        selection_tol=0.0,
        random_state=42,
        model_params=None,
        confidence_threshold=None,
        compute_validation=True,
    ):
        self.target = target
        self.features = features
        self.feature_selection = feature_selection
        self.cv = cv
        self.scoring = scoring
        self.class_weight = class_weight
        self.min_features = min_features
        self.selection_tol = selection_tol
        self.random_state = random_state
        self.model_params = model_params
        self.confidence_threshold = confidence_threshold
        self.compute_validation = compute_validation

    @staticmethod
    def _check_frame(data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not data.columns.is_unique:
            raise ValueError("data must have unique column names")

    def fit(self, data, groups=None):
        """Select predictors and fit on observed targets; groups align by row position.

        Pass a participant/household identifier array to keep related records
        together during selection. Missing group identifiers are rejected.
        """
        # A failed refit must not leave a previous model available for use.
        self.__dict__.pop("model_", None)
        self.__dict__.pop("imputation_summary_", None)
        self.oof_predictions_ = None
        self._check_frame(data)
        self._validate_threshold(self.confidence_threshold)
        if not isinstance(self.compute_validation, bool):
            raise ValueError("compute_validation must be True or False")
        if isinstance(self.features, str) or self.features is None:
            raise ValueError(
                "features must be an explicit non-empty sequence of columns"
            )
        features = list(self.features)
        if not features or len(set(features)) != len(features):
            raise ValueError("features must be non-empty and contain no duplicates")
        if self.target in features:
            raise ValueError("target must not appear in features")
        missing = [c for c in [self.target] + features if c not in data]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if self.feature_selection not in (None, "backward"):
            raise ValueError("feature_selection must be 'backward' or None")
        if self.scoring not in ("roc_auc_ovr", "accuracy", "f1_macro"):
            raise ValueError("scoring must be 'roc_auc_ovr', 'accuracy', or 'f1_macro'")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        if not isinstance(self.cv, numbers.Integral) or self.cv < 2:
            raise ValueError("cv must be an integer >= 2")
        if not isinstance(self.min_features, numbers.Integral) or self.min_features < 1:
            raise ValueError("min_features must be a positive integer")
        if not np.isfinite(self.selection_tol) or self.selection_tol < 0:
            raise ValueError("selection_tol must be finite and non-negative")
        params = dict(self.model_params or {})
        if any(
            params.get(p) is not None for p in ("early_stopping_rounds", "callbacks")
        ):
            raise ValueError(
                "Early stopping/callbacks require an evaluation set and are unsupported"
            )

        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                'Install imputation dependencies: pip install "cntmosaic[imputation]"'
            ) from exc
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        observed = data[self.target].notna().to_numpy()
        known = data.loc[observed]
        # Factorization preserves original labels instead of converting them to strings.
        y, classes = pd.factorize(known[self.target], sort=False)
        if len(classes) < 2:
            raise ValueError("At least two observed target classes are required")
        self.classes_ = np.asarray(classes)
        self.excluded_features_ = [c for c in features if known[c].isna().all()]
        features = [c for c in features if c not in self.excluded_features_]
        if not features or self.min_features > len(features):
            raise ValueError("Not enough non-empty predictors for min_features")
        if self.excluded_features_:
            warnings.warn(
                f"Excluded all-missing predictors: {self.excluded_features_}",
                UserWarning,
            )
        # Fix numeric/category roles on the observed training data. Category
        # vocabularies and numeric medians are still learned within each fold.
        self.numeric_features_ = (
            known[features].select_dtypes(include=np.number).columns.tolist()
        )

        def make_preprocessor(columns, frame):
            numeric = [c for c in columns if c in self.numeric_features_]
            categorical = [c for c in columns if c not in numeric]
            # Fold-empty numeric columns get a deterministic zero fallback.
            # This avoids dropping columns on older scikit-learn versions.
            empty = [c for c in numeric if frame[c].isna().all()]
            ordinary = [c for c in numeric if c not in empty]
            transformers = []
            if ordinary:
                transformers.append(
                    ("numeric", SimpleImputer(strategy="median"), ordinary)
                )
            if empty:
                transformers.append(
                    (
                        "empty_numeric",
                        FunctionTransformer(np.nan_to_num, kw_args={"nan": 0.0}),
                        empty,
                    )
                )
            if categorical:
                transformers.append(
                    (
                        "categorical",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        categorical,
                    )
                )
            return ColumnTransformer(transformers)

        def make_model():
            config = {
                "random_state": self.random_state,
                "n_jobs": 1,
                "eval_metric": "logloss" if len(classes) == 2 else "mlogloss",
            }
            config.update(params)
            return XGBClassifier(**config)

        def weights(labels):
            return (
                compute_sample_weight("balanced", labels) if self.class_weight else None
            )

        x = self._prepare_frame(known, features)
        group_values = None
        if groups is not None:
            group_values = np.asarray(groups)
            if group_values.ndim != 1 or len(group_values) != len(data):
                raise ValueError(
                    "groups must have one value per input row (positional order)"
                )
            group_values = group_values[observed]
            if pd.isna(group_values).any():
                raise ValueError(
                    "groups must not contain missing identifiers on observed rows"
                )

        self.selection_history_ = []
        self.baseline_score_ = self.selection_score_ = None
        current = features.copy()
        if self.feature_selection == "backward" or self.compute_validation:
            if group_values is None:
                if np.bincount(y).min() < self.cv:
                    raise ValueError(
                        "Each class needs at least cv observed rows; reduce cv"
                    )
                splitter = StratifiedKFold(
                    self.cv, shuffle=True, random_state=self.random_state
                )
            else:
                from sklearn.model_selection import StratifiedGroupKFold

                if len(pd.unique(group_values)) < self.cv:
                    raise ValueError("At least cv distinct groups are required")
                splitter = StratifiedGroupKFold(
                    self.cv, shuffle=True, random_state=self.random_state
                )
            folds = list(splitter.split(x, y, group_values))
            prepared = []
            for tr, va in folds:
                if len(np.unique(y[tr])) != len(classes) or len(
                    np.unique(y[va])
                ) != len(classes):
                    raise ValueError(
                        "Each CV train/validation fold must contain every class; reduce cv or revise groups"
                    )
                prep = make_preprocessor(features, x.iloc[tr])
                train = prep.fit_transform(x.iloc[tr])
                valid = prep.transform(x.iloc[va])
                # ColumnTransformer groups columns; record its actual output order.
                order = [
                    c
                    for name, _, cols in prep.transformers_
                    if name != "remainder"
                    for c in cols
                ]
                prepared.append((train, valid, tr, va, order))

            def evaluate(columns, return_probabilities=False):
                scores = []
                probabilities = np.empty((len(y), len(classes)))
                for train, valid, tr, va, order in prepared:
                    indices = [order.index(c) for c in columns]
                    model = make_model()
                    model.fit(train[:, indices], y[tr], sample_weight=weights(y[tr]))
                    proba = model.predict_proba(valid[:, indices])
                    probabilities[va] = proba
                    if self.scoring == "roc_auc_ovr":
                        score = (
                            roc_auc_score(y[va], proba[:, 1])
                            if len(classes) == 2
                            else roc_auc_score(
                                y[va], proba, multi_class="ovr", average="macro"
                            )
                        )
                    else:
                        pred = model.predict(valid[:, indices])
                        score = (
                            accuracy_score(y[va], pred)
                            if self.scoring == "accuracy"
                            else f1_score(y[va], pred, average="macro", zero_division=0)
                        )
                    scores.append(score)
                score = float(np.mean(scores))
                return (score, probabilities) if return_probabilities else score

        if self.feature_selection == "backward":
            best = self.baseline_score_ = evaluate(current)
            while len(current) > self.min_features:
                candidates = [
                    (evaluate([c for c in current if c != feature]), feature)
                    for feature in current
                ]
                score, feature = max(candidates, key=lambda item: item[0])
                if score < best - self.selection_tol:
                    break
                self.selection_history_.append(
                    {
                        "step": len(self.selection_history_) + 1,
                        "dropped": feature,
                        "score": score,
                        "delta": score - best,
                    }
                )
                current.remove(feature)
                best = score
            self.selection_score_ = best

        if self.compute_validation:
            _, probabilities = evaluate(current, return_probabilities=True)
            predicted = probabilities.argmax(axis=1)
            self.oof_predictions_ = pd.DataFrame(
                {
                    "row_position": np.flatnonzero(observed),
                    "true_label": self.classes_[y],
                    "predicted_label": self.classes_[predicted],
                    "confidence": probabilities.max(axis=1),
                    "correct": predicted == y,
                },
                index=known.index.copy(),
            )
            self.oof_probabilities_ = pd.DataFrame(
                probabilities, index=known.index.copy(), columns=self.classes_
            )
        else:
            self.oof_probabilities_ = None
        self.validation_note_ = (
            "Out-of-fold reference diagnostics, not independent validation. "
            "Feature selection reused these folds; results may be optimistic. "
            "Probabilities are uncalibrated and observed-target performance may "
            "not represent records with missing targets."
            if self.feature_selection == "backward"
            else "Out-of-fold reference diagnostics for fixed features. Probabilities "
            "are uncalibrated and observed-target performance may not represent "
            "records with missing targets."
        )
        self.selected_features_ = current
        self.preprocessor_ = make_preprocessor(current, x)
        train = self.preprocessor_.fit_transform(x[current])
        model = make_model()
        model.fit(train, y, sample_weight=weights(y))
        self.target_ = self.target
        self.model_ = model
        return self

    def _prepare_frame(self, data, features):
        frame = data[features].copy()
        for column in features:
            if column in self.numeric_features_:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
                    float
                )
                if np.isinf(frame[column].to_numpy()).any():
                    raise ValueError(f"Predictor {column!r} contains infinity")
            else:
                # Prefix real values so a literal missing-marker string cannot
                # collide with the categorical missing-value representation.
                frame[column] = (
                    frame[column]
                    .astype(object)
                    .map(
                        lambda value: (
                            "missing:" if pd.isna(value) else "value:" + str(value)
                        )
                    )
                )
        return frame

    @staticmethod
    def _validate_threshold(threshold):
        if threshold is not None and (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, numbers.Real)
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("confidence_threshold must be None or a number in [0, 1]")
        return threshold

    def _check_prediction_data(self, data, require_target=False):
        if not hasattr(self, "model_"):
            raise ValueError("Call fit before predicting or generating reports")
        self._check_frame(data)
        required = self.selected_features_ + ([self.target_] if require_target else [])
        missing = [c for c in required if c not in data]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def predict_proba(self, data):
        """Return class probabilities for every supplied row.

        Columns follow classes_; row order and duplicate indices are preserved.
        These are uncalibrated model probabilities. A target column is optional.
        """
        self._check_prediction_data(data)
        if data.empty:
            probabilities = np.empty((0, len(self.classes_)))
        else:
            frame = self._prepare_frame(data, self.selected_features_)
            probabilities = self.model_.predict_proba(
                self.preprocessor_.transform(frame)
            )
        return pd.DataFrame(
            probabilities, index=data.index.copy(), columns=self.classes_
        )

    def prediction_report(self, data):
        """Preview missing-target predictions without filling or filtering.

        Returns predicted_label, confidence and zero-based row_position, with
        the original index. row_position disambiguates duplicate index labels.
        Use predict_proba for the full per-class probability distribution.
        """
        self._check_prediction_data(data, require_target=True)
        positions = np.flatnonzero(data[self.target_].isna().to_numpy())
        probabilities = self.predict_proba(data.iloc[positions]).to_numpy()
        return pd.DataFrame(
            {
                "row_position": positions,
                "predicted_label": self.classes_[probabilities.argmax(axis=1)],
                "confidence": probabilities.max(axis=1),
            },
            index=data.index[positions].copy(),
        )

    @classmethod
    def _thresholds(cls, thresholds):
        values = list(thresholds)
        if not values:
            raise ValueError("thresholds must not be empty")
        for value in values:
            cls._validate_threshold(value)
        return values

    def confidence_report(self, data, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
        """Count missing targets filled/remaining at each threshold, without mutation.

        coverage is a fraction of the original missing-target records. It is
        zero for no missing records. Threshold 0 (or None) accepts all records.
        This report measures coverage, not correctness.
        """
        thresholds = self._thresholds(thresholds)
        predictions = self.prediction_report(data)
        confidence = predictions["confidence"].to_numpy()
        total = len(confidence)
        rows = []
        for threshold in thresholds:
            filled = int(
                np.count_nonzero(confidence >= (0 if threshold is None else threshold))
            )
            rows.append(
                {
                    "threshold": threshold,
                    "n_missing": total,
                    "n_imputed": filled,
                    "n_remaining": total - filled,
                    "coverage": filled / total if total else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def validation_report(self, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
        """Return threshold diagnostics from observed-target out-of-fold predictions.

        Returns a dict with summary and by_class DataFrames and a note string.
        Summary accuracy is correct/accepted. By-class coverage and accuracy
        condition on the true class; recall is correct/all true-class records
        (abstentions count as missed). Precision is correct/accepted predictions
        of that class. Undefined ratios are NaN, not zero. No holdout is reserved.
        Feature selection can make these reference results optimistic.
        """
        if not hasattr(self, "model_"):
            raise ValueError("Call fit before validation_report")
        if self.oof_predictions_ is None:
            raise ValueError(
                "Fit with compute_validation=True to enable validation_report"
            )
        thresholds = self._thresholds(thresholds)
        frame = self.oof_predictions_
        confidence = frame["confidence"].to_numpy()
        correct = frame["correct"].to_numpy()
        truth = frame["true_label"].to_numpy()
        predicted = frame["predicted_label"].to_numpy()
        summary, by_class = [], []
        for threshold in thresholds:
            accepted = confidence >= (0 if threshold is None else threshold)
            count = int(accepted.sum())
            n_correct = int((correct & accepted).sum())
            summary.append(
                {
                    "threshold": threshold,
                    "n_observed": len(frame),
                    "n_accepted": count,
                    "n_rejected": len(frame) - count,
                    "coverage": count / len(frame),
                    "n_correct": n_correct,
                    "accuracy": n_correct / count if count else np.nan,
                }
            )
            for label in self.classes_:
                true_class = truth == label
                support = int(true_class.sum())
                retained = int((true_class & accepted).sum())
                hits = int((true_class & accepted & correct).sum())
                predicted_count = int(((predicted == label) & accepted).sum())
                by_class.append(
                    {
                        "threshold": threshold,
                        "class_label": label,
                        "support": support,
                        "n_accepted": retained,
                        "n_correct": hits,
                        "coverage": retained / support,
                        "accuracy": hits / retained if retained else np.nan,
                        "recall": hits / support,
                        "n_predicted": predicted_count,
                        "precision": (
                            hits / predicted_count if predicted_count else np.nan
                        ),
                    }
                )
        return {
            "summary": pd.DataFrame(summary),
            "by_class": pd.DataFrame(by_class),
            "note": self.validation_note_,
        }

    def transform(self, data, confidence_threshold=_DEFAULT_THRESHOLD):
        """Fill missing targets above a threshold; return a copy.

        Omit confidence_threshold to use the constructor setting. Explicit None
        overrides that setting to fill all missing values. Threshold comparison
        is inclusive (>=). imputation_summary_ records counts from this call.
        """
        threshold = (
            self.confidence_threshold
            if confidence_threshold is _DEFAULT_THRESHOLD
            else confidence_threshold
        )
        self._validate_threshold(threshold)
        predictions = self.prediction_report(data)
        accepted = predictions["confidence"].to_numpy() >= (
            0 if threshold is None else threshold
        )
        positions = predictions["row_position"].to_numpy()[accepted]
        prediction = predictions["predicted_label"].to_numpy()[accepted]
        result = data.copy(deep=True)
        if len(positions):
            if isinstance(result[self.target_].dtype, pd.CategoricalDtype):
                extra = [
                    c
                    for c in self.classes_
                    if c not in result[self.target_].cat.categories
                ]
                if extra:
                    result[self.target_] = result[self.target_].cat.add_categories(
                        extra
                    )
            result.iloc[positions, result.columns.get_loc(self.target_)] = prediction
        total, filled = len(predictions), len(positions)
        self.imputation_summary_ = {
            "threshold": threshold,
            "n_missing": total,
            "n_imputed": filled,
            "n_remaining": total - filled,
            "coverage": filled / total if total else 0.0,
        }
        return result

    def fit_transform(self, data, groups=None, confidence_threshold=_DEFAULT_THRESHOLD):
        """Fit on observed targets, then fill missing values with optional filtering."""
        return self.fit(data, groups=groups).transform(
            data, confidence_threshold=confidence_threshold
        )
