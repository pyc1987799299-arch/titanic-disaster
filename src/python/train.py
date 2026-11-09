"""Train, evaluate, and export Titanic survival predictions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Age", "Fare", "FamilySize"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
REQUIRED = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def validate(frame, name, labeled=False):
    required = REQUIRED + (["Survived"] if labeled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{name}: no passenger rows")
    ids = pd.to_numeric(frame["PassengerId"], errors="raise")
    if (ids.isna().any() or not np.isfinite(ids).all() or ids.duplicated().any()
            or (ids < 1).any() or (ids > 1_000_000).any() or (ids % 1 != 0).any()):
        raise ValueError(f"{name}: PassengerId must be unique integers from 1 to 1000000")
    frame["PassengerId"] = ids.astype("int64")
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        values = pd.to_numeric(frame[col], errors="raise")
        if (values.dropna() < 0).any() or not np.isfinite(values.dropna()).all():
            raise ValueError(f"{name}: {col} must be nonnegative finite numbers or missing")
        frame[col] = values
    for col, allowed in {"Pclass": [1, 2, 3], "Sex": ["female", "male"],
                         "Embarked": ["C", "Q", "S"]}.items():
        if not frame[col].dropna().isin(allowed).all():
            raise ValueError(f"{name}: invalid {col} value")
    if labeled:
        if frame["Survived"].isna().any() or not frame["Survived"].isin([0, 1]).all():
            raise ValueError(f"{name}: Survived must contain only 0 or 1")
        if len(frame["Survived"].unique()) != 2 or frame["Survived"].value_counts().min() < 2:
            raise ValueError(f"{name}: need at least two passengers in each outcome class")


def features(frame):
    result = frame[["Age", "Fare", "Pclass", "Sex", "Embarked"]].copy()
    result["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1
    return result


def validation_mask(frame):
    # Identical stratified split in Python and R, independent of CSV row order.
    score = (frame["PassengerId"] * 1103515245 + 42) % 2147483648
    mask = pd.Series(False, index=frame.index)
    for outcome in (0, 1):
        group = frame.loc[frame["Survived"] == outcome, ["PassengerId"]].copy()
        group["score"] = score.loc[group.index]
        count = max(1, int(len(group) * 0.2))
        indices = group.sort_values(["score", "PassengerId"]).index[:count]
        mask.loc[indices] = True
    return mask


def build_model():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC),
                                    ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocess),
                     ("model", LogisticRegression(max_iter=2000, random_state=42))])


def describe_preprocessing(model):
    pre = model.named_steps["preprocess"]
    numeric = pre.named_transformers_["numeric"].named_steps["impute"].statistics_
    categorical = pre.named_transformers_["categorical"].named_steps["impute"].statistics_
    print("Numeric medians:", dict(zip(NUMERIC, map(float, numeric))))
    print("Categorical modes:", dict(zip(CATEGORICAL, categorical)))
    print("Numeric features standardized; categorical features one-hot encoded.")


def run(data_dir, output_dir):
    for name in ("train.csv", "test.csv"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Missing {data_dir / name}. Download the Kaggle files; see README.")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    validate(train, "train.csv", labeled=True)
    validate(test, "test.csv")
    if set(train["PassengerId"]) & set(test["PassengerId"]):
        raise ValueError("train.csv and test.csv contain overlapping PassengerId values")
    print(f"Loaded {len(train)} training passengers and {len(test)} test passengers.")
    print("Training missing values:\n" + train[REQUIRED].isna().sum().to_string())
    print("Test missing values:\n" + test[REQUIRED].isna().sum().to_string())
    print(f"Overall training survival rate: {train.Survived.mean():.4f}")
    for column in ("Sex", "Pclass"):
        print(f"Training survival rate by {column}:\n" + train.groupby(column).Survived.mean().to_string())
    print("Created FamilySize = SibSp + Parch + 1. Missing components yield missing FamilySize.")
    print("Features:", ", ".join(NUMERIC + CATEGORICAL))
    print("PassengerId is used only for splitting/export. Name, Ticket, and Cabin are excluded.")
    x, y = features(train), train["Survived"]
    mask = validation_mask(train)
    fit_x, fit_y = x.loc[~mask], y.loc[~mask]
    val_x, val_y = x.loc[mask], y.loc[mask]
    if fit_x.isna().all().any():
        raise ValueError("A feature is entirely missing in the fitting subset")
    print(f"Stratified holdout: {len(fit_x)} fitting rows, {len(val_x)} validation rows.")
    model = build_model().fit(fit_x, fit_y)
    print("Preprocessing learned only from the fitting subset:")
    describe_preprocessing(model)
    fit_accuracy = accuracy_score(fit_y, model.predict(fit_x))
    val_predictions = model.predict(val_x)
    val_accuracy = accuracy_score(val_y, val_predictions)
    baseline = float((val_y == fit_y.mode().iloc[0]).mean())
    print(f"Fitting-subset accuracy: {fit_accuracy:.4f}")
    print(f"Holdout validation accuracy: {val_accuracy:.4f}")
    print(f"Majority-class holdout baseline: {baseline:.4f}")
    print("Validation confusion matrix (rows actual 0/1, columns predicted 0/1):")
    print(confusion_matrix(val_y, val_predictions, labels=[0, 1]))
    final = build_model().fit(x, y)
    print("Refitted preprocessing and logistic regression on all labeled training rows:")
    describe_preprocessing(final)
    full_accuracy = accuracy_score(y, final.predict(x))
    print(f"Full-training accuracy (in-sample): {full_accuracy:.4f}")
    predictions = final.predict(features(test)).astype(int)
    print("Official test accuracy cannot be computed locally: test.csv has no true Survived labels.")
    print("gender_submission.csv contains example predictions, not ground truth.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions}).to_csv(
        output_dir / "submission.csv", index=False)
    fingerprint = hashlib.md5((output_dir / "submission.csv").read_bytes(), usedforsecurity=False).hexdigest()
    if fingerprint == "7996bcbd28372327caca4794088d0be6":
        print("Recorded Kaggle score for this exact submission file: 0.76794 (76.794%).")
        print("Source: completed manual Kaggle submission; this score was not computed in this run.")
    else:
        print("No recorded Kaggle score for this output file. Submit it to Kaggle for external scoring.")
    metrics = {"fit_rows": len(fit_x), "validation_rows": len(val_x),
               "train_rows": len(train), "test_rows": len(test),
               "fit_accuracy": fit_accuracy, "validation_accuracy": val_accuracy,
               "baseline_accuracy": baseline, "full_train_accuracy": full_accuracy}
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    print(f"Saved {len(predictions)} predictions to {output_dir / 'submission.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.csv'}. No passenger rows dropped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("src/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/python"))
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
