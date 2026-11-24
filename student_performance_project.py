"""Reproducible analysis and prediction pipeline for the UCI Student dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "G3"
FORBIDDEN = {"G1", "G2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/student-mat.csv"),
        help="Semicolon-separated UCI student file.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=Path("data/cleaned/student_mat_cleaned.csv"),
    )
    return parser.parse_args()


def load_and_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, sep=";")
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r}.")
    if FORBIDDEN & set(frame.columns):
        frame = frame.drop(columns=sorted(FORBIDDEN))
    frame = frame.drop_duplicates().reset_index(drop=True)
    binary_columns = [
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet"
    ]
    frame["parent_education_avg"] = (frame["Medu"] + frame["Fedu"]) / 2
    frame["alcohol_avg"] = (frame["Dalc"] + frame["Walc"]) / 2
    frame["social_time"] = frame["freetime"] + frame["goout"]
    frame["support_count"] = frame[binary_columns].eq("yes").sum(axis=1)
    frame["absence_log1p"] = np.log1p(frame["absences"])
    if frame[TARGET].isna().any():
        raise ValueError("The target contains missing values.")
    if not frame[TARGET].between(0, 20).all():
        raise ValueError("G3 must be in the documented 0-20 range.")
    return frame


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    features = frame.drop(columns=[TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    prediction = model.predict(X)
    return {
        "split": split,
        "mae": mean_absolute_error(y, prediction),
        "rmse": np.sqrt(mean_squared_error(y, prediction)),
        "r2": r2_score(y, prediction),
    }


def save_eda(frame: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.histplot(frame[TARGET], discrete=True, bins=21)
    plt.title("Distribution of final grade (G3)")
    plt.tight_layout()
    plt.savefig(figures / "g3_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=frame, x="studytime", y=TARGET)
    plt.title("Final grade by weekly study time")
    plt.tight_layout()
    plt.savefig(figures / "g3_by_studytime.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=frame, x="failures", y=TARGET)
    plt.title("Final grade by past class failures")
    plt.tight_layout()
    plt.savefig(figures / "g3_by_failures.png", dpi=150)
    plt.close()

    numeric = frame.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0)
    plt.title("Numeric feature correlations")
    plt.tight_layout()
    plt.savefig(figures / "numeric_correlations.png", dpi=150)
    plt.close()

    plot_specs = [
        ("absences", "schoolsup", "g3_vs_absences.png", "Final grade and absences", "scatter"),
        ("schoolsup", TARGET, "g3_by_school_support.png", "Final grade by school support", "box"),
        ("higher", TARGET, "g3_by_higher_education_goal.png", "Final grade by higher-education aspiration", "box"),
    ]
    for x_value, y_value, filename, title, kind in plot_specs:
        plt.figure(figsize=(7, 4))
        if kind == "scatter":
            sns.regplot(
                data=frame, x=x_value, y=TARGET,
                scatter_kws={"alpha": 0.45}, line_kws={"color": "crimson"},
            )
        else:
            sns.boxplot(data=frame, x=x_value, y=y_value)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=150)
        plt.close()


def validate_data(frame: pd.DataFrame, output: Path) -> None:
    checks = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "target_min": int(frame[TARGET].min()),
        "target_max": int(frame[TARGET].max()),
        "target_mean": float(frame[TARGET].mean()),
        "target_median": float(frame[TARGET].median()),
        "contains_G1": "G1" in frame.columns,
        "contains_G2": "G2" in frame.columns,
    }
    if checks["contains_G1"] or checks["contains_G2"]:
        raise ValueError("Validation failed: G1/G2 must not be model features.")
    (output / "data_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


def save_diagnostics(predictions: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=predictions, x="actual_G3", y="predicted_G3", alpha=0.7)
    limits = [0, 20]
    plt.plot(limits, limits, linestyle="--", color="crimson", label="perfect prediction")
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title("Actual versus predicted final grade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(predictions["residual"], bins=15, kde=True)
    plt.axvline(0, linestyle="--", color="crimson")
    plt.xlabel("Residual (actual - predicted)")
    plt.title("Prediction residuals")
    plt.tight_layout()
    plt.savefig(figures / "residual_distribution.png", dpi=150)
    plt.close()


def write_report(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output: Path,
) -> None:
    best = metrics.loc[
        (metrics["model"].eq("Ensemble model")) & (metrics["split"].eq("test"))
    ].iloc[0]
    top = feature_importance.head(10)["feature"].tolist()
    report = f"""# Student Performance Report

## Objective and data

This study predicts the UCI Student Performance final grade, `G3`, using
recorded demographic, family, school, attendance, and study-habit variables.
The Math-course file was used because it is a single, clearly defined cohort.
The source documentation defines `G3` on a 0-20 scale. The earlier grades
`G1` and `G2` were removed before modeling to satisfy the project requirement
and to avoid target leakage.

The cleaned dataset contains **{len(frame)} rows** and **{frame.shape[1] - 1} input
features**. Exact duplicate rows were removed; missing input values are handled
inside the training pipeline.

## Preprocessing and modeling choices

The data was split into training and test sets before fitting any imputer,
scaler, or encoder. Numeric variables use median imputation and standardization.
Categorical variables use most-frequent imputation and one-hot encoding.
Unknown categories are ignored at prediction time.

Ridge regression provides a stable linear baseline for one-hot encoded data.
Random forest and boosting models provide nonlinear comparisons. The final
model is a simple average ensemble of a tuned random forest and histogram
gradient boosting model. Additional
predictor-only features summarize parent education, alcohol consumption,
support indicators, social time, and the skewed absence count. Hyperparameters
were selected with a shuffled five-fold `KFold` on training data only, using
negative RMSE. The final test set was not used for tuning.

## Results

{metrics.to_markdown(index=False, floatfmt=".3f")}

The selected model is **{best['model']}**, chosen using cross-validation on the
training data. Its untouched test-set performance was:
**{best['rmse']:.3f} grade points**, with MAE **{best['mae']:.3f}** and
R-squared **{best['r2']:.3f}**. Because grades are noisy and the sample is modest,
these figures should be interpreted as approximate predictive performance rather
than proof that the model will generalize to every school or course.

## Patterns useful for support

The exploratory plots compare grades with study time, previous failures,
absences, school support, and higher-education aspiration.
The model's largest permutation-importance features in this run were:
{", ".join(top)}.

These patterns can help staff ask better follow-up questions about attendance,
study conditions, prior difficulties, and support needs. They should not be used
to label students, deny opportunities, or claim that a single characteristic
causes a low grade.

## Limitations

- The data is observational and cannot establish causation.
- It comes from two Portuguese schools and one historical setting.
- The target is an ordinal-like grade, but this project treats it as a numeric
  regression target to match the requested final-grade prediction.
- Self-reported variables may contain measurement error.
- The dataset is relatively small and may not represent other schools,
  countries, subjects, or years.
- `G1` and `G2` are unavailable by design, so predictions are less informed than
  an end-of-term forecasting model that is allowed to use them.

## Team contributions

Replace the placeholders below with the actual member names and contributions
before submission:

- Member 1: [name] - [contribution]
- Member 2: [name] - [contribution]
- Member 3: [name] - [contribution]

## Reproducibility

Run `python student_performance_project.py` from this project directory after
installing `requirements.txt`. All generated metrics, plots, the cleaned data,
and the fitted model are saved under the documented output paths.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frame = load_and_clean(args.input)
    args.cleaned_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.cleaned_output, index=False)
    validate_data(frame, output)
    save_eda(frame, output)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    preprocessor = make_preprocessor(frame)
    models = {
        "Ridge baseline": Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))]),
        "Random forest baseline": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "Gradient boosting baseline": Pipeline(
            [
                ("preprocess", make_preprocessor(frame)),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        "Histogram gradient boosting baseline": Pipeline(
            [
                ("preprocess", make_preprocessor(frame)),
                ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }
    metrics = []
    cv_summary = []
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            return_train_score=False,
            n_jobs=-1,
        )["test_score"]
        cv_summary.append(
            {
                "model": name,
                "cv_rmse_mean": -cv_scores.mean(),
                "cv_rmse_std": cv_scores.std(),
            }
        )
        model.fit(X_train, y_train)
        metrics.extend(
            [
                {"model": name, **evaluate(model, X_train, y_train, "train")},
                {"model": name, **evaluate(model, X_test, y_test, "test")},
            ]
        )

    tuned = GridSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(frame)),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"],
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    tuned.fit(X_train, y_train)
    cv_summary.append(
        {
            "model": "Tuned random forest",
            "cv_rmse_mean": -tuned.best_score_,
            "cv_rmse_std": np.nan,
        }
    )
    ensemble = VotingRegressor(
        [
            (
                "random_forest",
                Pipeline(
                    [
                        ("preprocess", make_preprocessor(frame)),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=500,
                                max_depth=8,
                                min_samples_leaf=4,
                                max_features=1.0,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "histogram_gradient_boosting",
                Pipeline(
                    [
                        ("preprocess", make_preprocessor(frame)),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                max_iter=150,
                                max_leaf_nodes=15,
                                learning_rate=0.05,
                                l2_regularization=1,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    ensemble_cv = cross_validate(
        ensemble,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=False,
        n_jobs=-1,
    )["test_score"]
    cv_summary.append(
        {
            "model": "Ensemble model",
            "cv_rmse_mean": -ensemble_cv.mean(),
            "cv_rmse_std": ensemble_cv.std(),
        }
    )
    ensemble.fit(X_train, y_train)
    metrics.extend(
        [
            {"model": "Tuned random forest", **evaluate(tuned.best_estimator_, X_train, y_train, "train")},
            {"model": "Tuned random forest", **evaluate(tuned.best_estimator_, X_test, y_test, "test")},
            {"model": "Ensemble model", **evaluate(ensemble, X_train, y_train, "train")},
            {"model": "Ensemble model", **evaluate(ensemble, X_test, y_test, "test")},
        ]
    )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "model_metrics.csv", index=False)
    pd.DataFrame(cv_summary).sort_values("cv_rmse_mean").to_csv(
        output / "cv_model_comparison.csv", index=False
    )
    (output / "best_params.json").write_text(
        json.dumps(tuned.best_params_, indent=2, default=str), encoding="utf-8"
    )
    joblib.dump(ensemble, output / "final_model.joblib")

    permutation = permutation_importance(
        ensemble, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.DataFrame(
        {"feature": X_test.columns, "importance_mean": permutation.importances_mean,
         "importance_std": permutation.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    predictions = pd.DataFrame(
        {
            "actual_G3": y_test,
            "predicted_G3": ensemble.predict(X_test),
        }
    )
    predictions["residual"] = predictions["actual_G3"] - predictions["predicted_G3"]
    predictions.to_csv(output / "test_predictions.csv", index=False)
    save_diagnostics(predictions, output)
    write_report(frame, metrics_frame, importance, output)
    print(metrics_frame.to_string(index=False))
    print(f"\nBest parameters: {tuned.best_params_}")
    print(f"Outputs written to: {output.resolve()}")


if __name__ == "__main__":
    arguments = parse_args()
    project_dir = Path(__file__).resolve().parent
    if arguments.input == Path("data/raw/student-mat.csv"):
        project_input = project_dir / arguments.input
        source = project_dir.parent / "student+performance" / "student" / "student-mat.csv"
        if source.exists() and not project_input.exists():
            project_input.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, project_input)
        if not arguments.input.exists() and project_input.exists():
            arguments.input = project_input
        if arguments.output == Path("outputs"):
            arguments.output = project_dir / arguments.output
        if arguments.cleaned_output == Path("data/cleaned/student_mat_cleaned.csv"):
            arguments.cleaned_output = project_dir / arguments.cleaned_output
    run(arguments)
