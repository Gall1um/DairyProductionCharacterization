"""Core data validation and ML for the Dairy Yield Monitor.

The models estimate contemporaneous milk yield, NOT future milk yield or disease.
Cross-validation holds out whole cows. Error bands are empirical and approximate.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Milk Yield (L/day/cow)"
MILK_OUTPUTS = (
    TARGET,
    "Fat (%)", "Protein (%)", "Lactose (%)", "Solids-not-fat (%)",
    "Salt (%)", "Milk pH", "Somatic cell count (cells/mL)",
)
OUTPUT_DETAILS = {
    TARGET: ("Daily milk yield", "L/day", "Estimated milk volume per cow per day"),
    "Fat (%)": ("Milk fat", "%", "Percentage of milk that is fat"),
    "Protein (%)": ("Milk protein", "%", "Percentage of milk that is protein"),
    "Lactose (%)": ("Lactose", "%", "Milk sugar percentage"),
    "Solids-not-fat (%)": ("Solids-not-fat", "%", "Non-fat milk solids, including proteins, lactose, and minerals"),
    "Salt (%)": ("Salt", "%", "Salt percentage as reported by the laboratory"),
    "Milk pH": ("Milk pH", "pH", "Acidity/alkalinity scale, not a nutrient"),
    "Somatic cell count (cells/mL)": ("Somatic cell count", "cells/mL", "Milk quality indicator, not a diagnosis"),
}
OUTPUT_ALIASES = {
    "Fat%": "Fat (%)", "Fat_%": "Fat (%)", "Fat %": "Fat (%)",
    "Protein %": "Protein (%)", "Protein_%": "Protein (%)",
    "Lactose%": "Lactose (%)", "Lactose_%": "Lactose (%)",
    "SNF%": "Solids-not-fat (%)", "SNF_%": "Solids-not-fat (%)",
    "Salt%": "Salt (%)", "Salt_%": "Salt (%)",
    "pH": "Milk pH",
    "SCC cells per mL": "Somatic cell count (cells/mL)",
    "SCC_cells_per_mL": "Somatic cell count (cells/mL)",
}

ID = "Cow ID"
CATEGORICAL = ("THI", "Genetic group")
PHYSIOLOGY = ("Rectal temp (F)", "Pulse (bpm)", "Respiration (bpm)")
FEED = ("DMI (kg)",)
BLOOD = (
    "Glucose (mmol/L)", "Total Protein (g/dL)", "Uric Acid (mg/dL)",
    "Cholesterol (mg/dL)", "Calcium (mg/dL)", "HDL (mg/dL)",
    "AST (U/I)", "ALT (U/I)", "Cortisol (µg/dL)",
)
FEATURES = CATEGORICAL + PHYSIOLOGY + FEED + BLOOD
PANELS = {
    "THI + genetics": CATEGORICAL,
    "+ physiology": CATEGORICAL + PHYSIOLOGY,
    "+ feed intake": CATEGORICAL + FEED,
    "+ blood panel": CATEGORICAL + BLOOD,
    "Practical sensors": CATEGORICAL + PHYSIOLOGY + FEED,
    "All available measurements": FEATURES,
}

ALIASES = {
    **OUTPUT_ALIASES,
    "Animal ID": ID, "Cow_ID": ID, "cow_id": ID, "Cow ID": ID,
    "Genetic Group": "Genetic group", "Genetic_Group": "Genetic group",
    "THI Range": "THI", "THI_Range": "THI", "THI category": "THI",
    "Rectal Temp (F)": "Rectal temp (F)",
    "Pulse Rate (bpm)": "Pulse (bpm)",
    "Respiration Rate (bpm)": "Respiration (bpm)",
    "Dry_Matter_Intake_(DMI)_Kg_per_day": "DMI (kg)",
    "Milk_Yield_L_per_day": TARGET,
}


class DataProblem(ValueError):
    """Actionable validation error for the app UI."""


def norm_id(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def norm_thi(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text.startswith(('T0', 'T1', 'T2')):
        return text[:2]
    return text


def norm_genetics(value):
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper().replace('_', '').replace(' ', '')


def normalize_record_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(x).strip() for x in df.columns]
    renames = {key: val for key, val in ALIASES.items() if key in df.columns and val not in df.columns}
    df = df.rename(columns=renames)
    if df.columns.duplicated().any():
        raise DataProblem("The data has duplicate column names after normalization.")
    if ID not in df.columns:
        raise DataProblem("Add a 'Cow ID' column. Cow IDs are required for cow-grouped validation.")
    if TARGET not in df.columns:
        raise DataProblem(f"Add the target column '{TARGET}' to historical training data.")
    df[ID] = df[ID].map(norm_id)
    if "Observation date" in df.columns:
        parsed_dates = pd.to_datetime(df["Observation date"], errors="coerce")
        df["Observation date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    if "THI" in df:
        df["THI"] = df["THI"].map(norm_thi)
    if "Genetic group" in df:
        df["Genetic group"] = df["Genetic group"].map(norm_genetics)
    for col in FEATURES:
        if col in df and col not in CATEGORICAL:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for output in MILK_OUTPUTS:
        if output in df:
            df[output] = pd.to_numeric(df[output], errors="coerce")
    df = df.dropna(subset=[ID, TARGET]).reset_index(drop=True)
    if df.empty:
        raise DataProblem("There are no rows with both a Cow ID and a numeric milk yield.")
    if df[ID].nunique() < 3:
        raise DataProblem("At least three different cows are needed for meaningful grouped validation.")
    if not available_features(df):
        raise DataProblem("No supported predictor columns were found. Start with THI and genetic group.")
    return df


def _prep_original(df: pd.DataFrame, rename: dict) -> pd.DataFrame:
    df = df.rename(columns=rename).copy()
    df[ID] = df[ID].map(norm_id)
    df["THI"] = df["THI"].map(norm_thi)
    df["Genetic group"] = df["Genetic group"].map(norm_genetics)
    df["Replication No"] = pd.to_numeric(df["Replication No"], errors="raise").astype(int)
    keys = [ID, "THI", "Replication No"]
    if df.duplicated(keys).any():
        raise DataProblem("Duplicate (Cow ID, THI, replication) keys in a source sheet.")
    return df


def load_study_workbooks(directory: str | Path) -> tuple[pd.DataFrame, dict]:
    """Join the three provided files by exact recorded IDs, without inventing corrections.

    The apparent one-ID offset in physiology means these are PROVISIONAL joins.
    """
    directory = Path(directory)
    paths = {
        "physiology": directory / 'physiological responses.xlsx',
        "milk": directory / 'DMI, milk yield and composition.xlsx',
        "blood": directory / 'Blood metabolites.xlsx',
    }
    absent = [p.name for p in paths.values() if not p.exists()]
    if absent:
        raise FileNotFoundError('Missing study files in data/: ' + ', '.join(absent))
    p = _prep_original(pd.read_excel(paths['physiology']), {
        "Animal ID": ID, "THI Range": "THI", "Genetic group": "Genetic group",
        "Rectal Temp (F)": "Rectal temp (F)",
        "Pulse Rate (bpm)": "Pulse (bpm)",
        "Respiration Rate (bpm)": "Respiration (bpm)",
    })
    m = _prep_original(pd.read_excel(paths['milk']), {
        "Animal ID": ID, "THI Range": "THI", "Genetic Group": "Genetic group",
    })
    b = _prep_original(pd.read_excel(paths['blood']), {
        "Animal ID": ID, "THI Range": "THI", "Genetic Group": "Genetic group",
    })
    keys = [ID, "THI", "Replication No"]
    m = m.rename(columns=OUTPUT_ALIASES)
    m = m[keys + ["Genetic group", "DMI (kg)", *MILK_OUTPUTS]]
    p = p[keys + ["Genetic group", *PHYSIOLOGY]].rename(columns={"Genetic group": "Genetic group (physiology)"})
    b = b[keys + ["Genetic group", *BLOOD]].rename(columns={"Genetic group": "Genetic group (blood)"})
    merged = m.merge(b, on=keys, how="inner", validate="one_to_one")
    merged = merged.merge(p, on=keys, how="inner", validate="one_to_one")
    mismatched = ((merged["Genetic group"] != merged["Genetic group (blood)"]) |
                  (merged["Genetic group"] != merged["Genetic group (physiology)"]))
    if mismatched.any():
        raise DataProblem(f"{mismatched.sum()} joined records disagree on genetic group.")
    merged = merged.drop(columns=["Genetic group (blood)", "Genetic group (physiology)"])
    merged = normalize_record_frame(merged)
    meta = {
        "source_counts": {"physiology": len(p), "milk": len(m), "blood": len(b)},
        "joined_rows": len(merged), "joined_cows": merged[ID].nunique(),
        "source_warning": "The source physiology IDs appear offset by one relative to milk/blood. "
                          "Exact-key matches are provisional; model performance is illustrative "
                          "until animal identity is confirmed.",
    }
    return merged, meta


def read_farm_upload(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    source = BytesIO(file_bytes)
    try:
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(source)
        elif file_name.lower().endswith(".xlsx"):
            df = pd.read_excel(source)
        else:
            raise DataProblem("Upload a combined .csv or .xlsx historical file.")
    except Exception as exc:
        raise DataProblem(f"Could not read the uploaded dataset: {exc}") from exc
    return normalize_record_frame(df)


def available_features(df: pd.DataFrame) -> list[str]:
    return [name for name in FEATURES if name in df and df[name].notna().any()]


def make_pipeline(feature_names: Sequence[str], algorithm: str = "Ridge"):
    feature_names = tuple(feature_names)
    if not feature_names:
        return DummyRegressor(strategy="mean")
    if any(name not in FEATURES for name in feature_names):
        raise DataProblem("Unsupported feature name")
    cats = [name for name in feature_names if name in CATEGORICAL]
    nums = [name for name in feature_names if name not in CATEGORICAL]
    parts = []
    if cats:
        parts.append(("categories", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), cats))
    if nums:
        parts.append(("numerics", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), nums))
    prep = ColumnTransformer(parts, remainder="drop")
    if algorithm == "Ridge":
        estimator = Ridge(alpha=10.0)
    elif algorithm == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=130, max_depth=7, min_samples_leaf=3,
            random_state=42, n_jobs=-1)
    else:
        raise DataProblem(f"Unsupported model: {algorithm}")
    return Pipeline([("preprocess", prep), ("regressor", estimator)])


@dataclass
class ValidationResult:
    target: str
    pipeline: object
    features: tuple[str, ...]
    algorithm: str
    rows: int
    cows: int
    folds: int
    mae: float
    rmse: float
    r2: float
    fold_mae: tuple[float, ...]
    q80: float
    q90: float
    q95: float
    observed: np.ndarray
    oof_predictions: np.ndarray


def validate_and_train(
    df: pd.DataFrame, feature_names: Sequence[str], algorithm="Ridge", target=TARGET
) -> ValidationResult:
    feature_names = tuple(feature_names)
    if any(name not in available_features(df) for name in feature_names):
        raise DataProblem("A selected feature is missing from the historical dataset.")
    if target not in MILK_OUTPUTS:
        raise DataProblem("That milk outcome is not supported.")
    if target not in df.columns:
        raise DataProblem(f"No historical measurements of {target} were uploaded.")
    # For a fair sensor comparison, omit rows with missing selected measurements.
    # Never silently pretend a missing sensor reading was measured.
    subset = df.dropna(subset=[target, *feature_names]).reset_index(drop=True)
    if len(subset) < 15:
        raise DataProblem(f"Not enough complete historical observations to learn {target} with these inputs.")
    groups = subset[ID].to_numpy()
    y = subset[target].to_numpy(dtype=float)
    X = subset[list(feature_names)] if feature_names else pd.DataFrame(index=subset.index)
    folds = min(5, len(np.unique(groups)))
    if folds < 3:
        raise DataProblem("Need at least three cows for grouped validation.")
    oof = np.full(len(subset), np.nan)
    fold_mae = []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(X, y, groups):
        model = make_pipeline(feature_names, algorithm)
        model.fit(X.iloc[train_idx], y[train_idx])
        predictions = model.predict(X.iloc[test_idx])
        oof[test_idx] = predictions
        fold_mae.append(float(mean_absolute_error(y[test_idx], predictions)))
    assert np.isfinite(oof).all()
    absolute_errors = np.abs(oof - y)
    final_model = make_pipeline(feature_names, algorithm)
    final_model.fit(X, y)
    return ValidationResult(
        target=target, pipeline=final_model, features=feature_names, algorithm=algorithm,
        rows=len(subset), cows=int(len(np.unique(groups))), folds=folds,
        mae=float(mean_absolute_error(y, oof)),
        rmse=float(np.sqrt(mean_squared_error(y, oof))),
        r2=float(r2_score(y, oof)),
        fold_mae=tuple(fold_mae),
        q80=float(np.quantile(absolute_errors, 0.80)),
        q90=float(np.quantile(absolute_errors, 0.90)),
        q95=float(np.quantile(absolute_errors, 0.95)),
        observed=y.copy(), oof_predictions=oof.copy(),
    )


def predict_one(result: ValidationResult, record: dict) -> float:
    row = pd.DataFrame([{name: record.get(name, np.nan) for name in result.features}])
    if "THI" in row:
        row["THI"] = row["THI"].map(norm_thi)
    if "Genetic group" in row:
        row["Genetic group"] = row["Genetic group"].map(norm_genetics)
    for col in result.features:
        if col not in CATEGORICAL:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    value = float(result.pipeline.predict(row)[0])
    # Do not return negative volume, percentages, or counts.
    # pH and percentages require no other invented postprocessing.
    if result.target == "Milk pH":
        return value
    if result.target in ("Fat (%)", "Protein (%)", "Lactose (%)", "Solids-not-fat (%)", "Salt (%)"):
        return float(np.clip(value, 0, 100))
    return max(0.0, value)


def classify_deviation(actual: float, predicted: float, result: ValidationResult):
    """Rules flag unusual production deviations, NOT clinical health states."""
    delta = float(actual) - float(predicted)
    magnitude = abs(delta)
    if magnitude <= result.q80:
        return "GREEN", "Within typical prediction variation", delta
    if magnitude <= result.q95:
        return "YELLOW", "Larger-than-usual production deviation — review records", delta
    return "RED", "Unusually large production deviation — prioritize follow-up", delta


def template_csv_bytes() -> bytes:
    return (','.join([ID, "Observation date", *FEATURES, *MILK_OUTPUTS]) + '\n').encode('utf-8')
