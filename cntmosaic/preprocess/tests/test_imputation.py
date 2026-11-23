import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from cntmosaic.preprocess import XGBoostImputer


@pytest.fixture
def data():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 66).astype(float),
            "setting": pd.Categorical(["home", "work", "school"] * 22),
            "noise": rng.normal(size=66),
            "target": pd.Categorical(["low", "middle", "high"] * 22),
        }
    )
    frame.loc[60:, "target"] = np.nan
    frame.loc[[1, 61], "age"] = np.nan
    frame.index = [i // 2 for i in range(len(frame))]  # duplicate indices
    return frame


def make_imputer(**kwargs):
    config = dict(
        target="target",
        features=["age", "setting", "noise"],
        cv=2,
        model_params={"n_estimators": 4, "max_depth": 2},
    )
    config.update(kwargs)
    return XGBoostImputer(**config)


@pytest.mark.parametrize("selection", [None, "backward"])
@pytest.mark.parametrize("balanced", [None, "balanced"])
def test_imputation_preserves_observed_rows_and_input(data, selection, balanced):
    before = data.copy(deep=True)
    imputer = make_imputer(feature_selection=selection, class_weight=balanced)
    completed = imputer.fit_transform(data)
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(completed.iloc[:60], data.iloc[:60])
    assert completed.target.notna().all()
    assert completed.index.equals(data.index)
    assert set(completed.target) == set(data.target.cat.categories)
    assert set(imputer.selected_features_) <= set(imputer.features)
    assert (imputer.selection_score_ is None) == (selection is None)


def test_unseen_categories_and_fold_empty_numeric(data):
    data["empty_in_some_folds"] = np.nan
    data.iloc[0, data.columns.get_loc("empty_in_some_folds")] = 5
    imputer = make_imputer(features=["setting", "empty_in_some_folds"])
    imputer.fit(data)
    future = pd.DataFrame(
        {"target": [None], "setting": ["new setting"], "empty_in_some_folds": [np.nan]}
    )
    assert imputer.transform(future).target.notna().all()


def test_grouped_selection(data):
    # Each group contains all three classes, making grouped stratification feasible.
    groups = np.arange(len(data)) // 3
    imputer = make_imputer()
    result = imputer.fit_transform(data, groups=groups)
    assert result.target.notna().all()
    assert np.isfinite(imputer.selection_score_)


@pytest.mark.parametrize("scoring", ["roc_auc_ovr", "accuracy", "f1_macro"])
def test_binary_numeric_labels(data, scoring):
    frame = data.copy()
    frame["target"] = np.tile([10.0, 20.0], 33)
    frame.iloc[60:, frame.columns.get_loc("target")] = np.nan
    imputer = make_imputer(scoring=scoring)
    result = imputer.fit_transform(frame)
    assert set(result.target) == {10.0, 20.0}
    assert np.isfinite(imputer.selection_score_)


def test_all_missing_predictor_and_final_fit_uses_all_known_rows(data):
    data["empty"] = np.nan
    imputer = make_imputer(features=["age", "empty"], feature_selection=None)
    with pytest.warns(UserWarning, match="all-missing"):
        imputer.fit(data)
    assert imputer.excluded_features_ == ["empty"]
    expected = data.iloc[:60].age.median()
    np.testing.assert_allclose(
        imputer.preprocessor_.named_transformers_["numeric"].statistics_, [expected]
    )


def test_invalid_inputs_and_failed_refit(data):
    imputer = make_imputer(feature_selection=None).fit(data)
    imputer.features = ["target"]
    with pytest.raises(ValueError, match="target must not"):
        imputer.fit(data)
    with pytest.raises(ValueError, match="Call fit"):
        imputer.transform(data)
    with pytest.raises(ValueError, match="two observed"):
        make_imputer().fit(data.assign(target="one"))
    with pytest.raises(ValueError, match="one value per"):
        make_imputer().fit(data, groups=[1])
    with pytest.raises(ValueError, match="Each class"):
        make_imputer(cv=30).fit(data)


def test_transform_adds_known_labels_to_new_categorical_schema(data):
    imputer = make_imputer(feature_selection=None).fit(data)
    future = data.iloc[-2:].copy()
    future["target"] = pd.Categorical([None, None], categories=["low"])
    result = imputer.transform(future)
    assert result.target.notna().all()
    assert set(result.target.cat.categories) == set(imputer.classes_)


def test_threshold_reports_and_inclusive_filtering(data, monkeypatch):
    imputer = make_imputer(feature_selection=None, confidence_threshold=0.8).fit(data)
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    probabilities = np.column_stack(
        [confidence, (1 - confidence) / 2, (1 - confidence) / 2]
    )
    monkeypatch.setattr(
        imputer,
        "predict_proba",
        lambda rows: pd.DataFrame(
            probabilities, index=rows.index, columns=imputer.classes_
        ),
    )
    before = data.copy(deep=True)
    preview = imputer.prediction_report(data)
    assert preview.row_position.tolist() == list(range(60, 66))
    np.testing.assert_allclose(preview.confidence, confidence)
    report = imputer.confidence_report(data, thresholds=[None, 0.8, 1.0])
    assert report.n_imputed.tolist() == [6, 2, 0]
    assert report.n_remaining.tolist() == [0, 4, 6]
    result = imputer.transform(data)
    assert result.target.isna().sum() == 4
    pd.testing.assert_frame_equal(result.iloc[:60], data.iloc[:60])
    assert imputer.imputation_summary_["n_imputed"] == 2
    assert imputer.transform(data, confidence_threshold=None).target.notna().all()
    pd.testing.assert_frame_equal(data, before)


def test_oof_validation_threshold_metrics(data):
    imputer = make_imputer().fit(data, groups=np.arange(len(data)) // 3)
    oof = imputer.oof_predictions_
    assert len(oof) == 60
    assert oof.row_position.tolist() == list(range(60))
    np.testing.assert_allclose(imputer.oof_probabilities_.sum(axis=1), 1.0, atol=1e-6)
    report = imputer.validation_report(thresholds=[None, 0.8, 1.0])
    all_rows = report["summary"].iloc[0]
    assert all_rows.n_accepted == 60
    assert all_rows.accuracy == pytest.approx(oof.correct.mean())
    assert "optimistic" in report["note"]
    for threshold, rows in report["by_class"].groupby("threshold", dropna=False):
        assert rows.support.sum() == 60
        selected = (
            report["summary"].threshold.isna()
            if pd.isna(threshold)
            else report["summary"].threshold == threshold
        )
        summary = report["summary"].loc[selected].iloc[0]
        assert rows.n_accepted.sum() == summary.n_accepted
        assert rows.n_correct.sum() == summary.n_correct
        assert rows.n_predicted.sum() == summary.n_accepted
    none_accepted = report["summary"].iloc[-1]
    assert none_accepted.n_accepted == 0
    assert pd.isna(none_accepted.accuracy)


def test_empty_predictions_and_no_missing_values(data):
    imputer = make_imputer(feature_selection=None).fit(data)
    known = data.iloc[:60]
    assert imputer.prediction_report(known).empty
    report = imputer.confidence_report(known)
    assert (report.n_missing == 0).all()
    assert (report.coverage == 0).all()
    pd.testing.assert_frame_equal(imputer.transform(known), known)
    assert imputer.predict_proba(known.iloc[:0]).shape == (0, 3)
    pd.testing.assert_frame_equal(imputer.transform(known.iloc[:0]), known.iloc[:0])


def test_optional_validation_and_fixed_feature_cv(data):
    imputer = make_imputer(feature_selection=None).fit(data)
    assert imputer.selection_score_ is None
    assert len(imputer.oof_predictions_) == 60
    assert "fixed features" in imputer.validation_report()["note"]
    # Without selection or diagnostics, small classes do not need CV support.
    small = data.iloc[[0, 1, 2, 60]].copy()
    imputer = make_imputer(feature_selection=None, compute_validation=False).fit(small)
    assert imputer.transform(small).target.notna().all()
    with pytest.raises(ValueError, match="compute_validation=True"):
        imputer.validation_report()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, "0.8", True])
def test_invalid_thresholds(data, threshold):
    with pytest.raises(ValueError, match="confidence_threshold"):
        make_imputer(confidence_threshold=threshold).fit(data)
    imputer = make_imputer(feature_selection=None, compute_validation=False).fit(data)
    with pytest.raises(ValueError, match="confidence_threshold"):
        imputer.transform(data, confidence_threshold=threshold)
    with pytest.raises(ValueError, match="confidence_threshold"):
        imputer.confidence_report(data, thresholds=[threshold])


def test_fit_transform_threshold_override(data):
    imputer = make_imputer(feature_selection=None, confidence_threshold=1.0)
    assert imputer.fit_transform(data, confidence_threshold=None).target.notna().all()
    assert imputer.imputation_summary_["threshold"] is None
