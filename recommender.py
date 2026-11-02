"""Personalized learning recommendation model for the GR-1 college project."""

from pathlib import Path
import json

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERACTIONS_PATH = DATA_DIR / "student_learning_interactions.csv"
RESOURCES_PATH = DATA_DIR / "learning_resources.csv"

MODEL_PATH = OUTPUT_DIR / "suitability_model.joblib"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
MATRIX_CSV_PATH = OUTPUT_DIR / "confusion_matrix.csv"
MATRIX_IMAGE_PATH = OUTPUT_DIR / "confusion_matrix.png"

# ============================================================
# FEATURES AND TARGET
# ============================================================

NUMERIC_FEATURES = [
    "performance_score",
    "topic_completed",
]

CATEGORICAL_FEATURES = [
    "interest_category",
    "learning_preference",
    "resource_category",
    "topic",
    "resource_type",
    "difficulty",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "suitable"


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

def load_data():
    """Load datasets, validate columns, and clean basic data issues."""

    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Interaction dataset not found: {INTERACTIONS_PATH}"
        )

    if not RESOURCES_PATH.exists():
        raise FileNotFoundError(
            f"Learning resources dataset not found: {RESOURCES_PATH}"
        )

    interactions = pd.read_csv(INTERACTIONS_PATH)
    resources = pd.read_csv(RESOURCES_PATH)

    required_columns = set(FEATURES + [TARGET])
    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"Interaction dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    interactions = interactions.drop_duplicates().copy()

    interactions["performance_score"] = pd.to_numeric(
        interactions["performance_score"],
        errors="coerce",
    ).clip(0, 100)

    interactions["topic_completed"] = pd.to_numeric(
        interactions["topic_completed"],
        errors="coerce",
    ).fillna(0).astype(int).clip(0, 1)

    interactions[TARGET] = pd.to_numeric(
        interactions[TARGET],
        errors="coerce",
    )

    interactions = interactions.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    interactions[TARGET] = interactions[TARGET].astype(int)

    # The model is a binary classifier: 0 or 1.
    invalid_labels = ~interactions[TARGET].isin([0, 1])

    if invalid_labels.any():
        raise ValueError(
            "The 'suitable' target column must contain only 0 and 1."
        )

    if interactions[TARGET].nunique() < 2:
        raise ValueError(
            "The dataset must contain both suitability classes: 0 and 1."
        )

    return interactions, resources


# ============================================================
# BUILD RANDOM FOREST PIPELINE
# ============================================================

def build_pipeline():
    """Build preprocessing and Random Forest classification pipeline."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model_pipeline = Pipeline([
        ("preprocess", preprocessing),
        (
            "model",
            RandomForestClassifier(
                n_estimators=160,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    return model_pipeline


# ============================================================
# TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate():
    """Train, evaluate, save metrics, and create evaluation charts."""

    data, _ = load_data()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    metrics = {
        "test_rows": int(len(y_test)),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    # Save evaluation metrics as JSON.
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "Not Suitable (0)",
            "Suitable (1)",
        ],
        zero_division=0,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Confusion matrix CSV
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            "Actual_Not_Suitable",
            "Actual_Suitable",
        ],
        columns=[
            "Predicted_Not_Suitable",
            "Predicted_Suitable",
        ],
    )

    matrix_dataframe.to_csv(MATRIX_CSV_PATH)

    # --------------------------------------------------------
    # Confusion matrix image
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Not Suitable (0)",
            "Suitable (1)",
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "Random Forest - Confusion Matrix",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    fig.tight_layout()

    fig.savefig(
        MATRIX_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    return model, metrics


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

def recommend(
    performance_score,
    interest_category,
    learning_preference,
    completed_topics=None,
    top_n=5,
    model=None,
):
    """
    Rank learning resources by predicted suitability.

    Completed topics are placed after uncompleted topics.
    Only one recommendation per topic is returned.
    """

    # Validate student performance score.
    performance_score = float(performance_score)

    if not 0 <= performance_score <= 100:
        raise ValueError(
            "Performance score must be between 0 and 100."
        )

    if int(top_n) < 1:
        raise ValueError(
            "The number of recommendations must be at least 1."
        )

    # Load or train the model.
    if model is None:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        else:
            model, _ = train_and_evaluate()

    _, resources = load_data()

    completed_topics = {
        str(topic).strip().casefold()
        for topic in (completed_topics or [])
        if str(topic).strip()
    }

    rows = []

    for _, resource in resources.iterrows():

        topic = str(resource["topic"])

        completed = int(
            topic.strip().casefold() in completed_topics
        )

        row = {
            "performance_score": performance_score,
            "topic_completed": completed,
            "interest_category": interest_category,
            "learning_preference": learning_preference,
            "resource_category": resource["category"],
            "topic": topic,
            "resource_type": resource["resource_type"],
            "difficulty": resource["difficulty"],
        }

        rows.append(row)

    candidates = pd.DataFrame(rows)

    # Generate predicted suitability probabilities.
    probabilities = model.predict_proba(
        candidates[FEATURES]
    )[:, 1]

    candidates["suitability_score"] = probabilities

    # Attach learning resource information.
    candidates["resource_id"] = resources[
        "resource_id"
    ].values

    candidates["estimated_minutes"] = resources[
        "estimated_minutes"
    ].values

    candidates["description"] = resources[
        "description"
    ].values

    candidates["url"] = resources["url"].values

    # Prioritize topics the student has not completed.
    candidates["completed_penalty"] = (
        candidates["topic"]
        .str.strip()
        .str.casefold()
        .isin(completed_topics)
        .astype(int)
    )

    candidates = candidates.sort_values(
        by=[
            "completed_penalty",
            "suitability_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    # Return only one resource per topic.
    candidates = candidates.drop_duplicates(
        subset=["topic"]
    )

    recommendations = candidates.head(
        int(top_n)
    ).drop(
        columns=["completed_penalty"]
    ).reset_index(drop=True)

    return recommendations


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    try:
        trained_model, evaluation_results = train_and_evaluate()

        print("\n" + "=" * 55)
        print("MODEL TRAINED AND EVALUATED SUCCESSFULLY")
        print("=" * 55)

        print("\nEvaluation Metrics:")
        print(
            pd.Series(evaluation_results).to_string()
        )

        print("\nClassification Report:")
        print(
            REPORT_PATH.read_text(
                encoding="utf-8"
            )
        )

        print("\nConfusion Matrix:")
        print(
            pd.read_csv(
                MATRIX_CSV_PATH,
                index_col=0,
            ).to_string()
        )

        print("\nGenerated Output Files:")
        print(f"Trained model:       {MODEL_PATH}")
        print(f"Evaluation metrics:  {METRICS_PATH}")
        print(f"Classification report: {REPORT_PATH}")
        print(f"Confusion matrix CSV:  {MATRIX_CSV_PATH}")
        print(f"Confusion matrix image: {MATRIX_IMAGE_PATH}")

        print("\nAll evaluation files have been saved.")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
        print(
            "Check that both CSV files exist and contain "
            "the required columns and valid data."
        )

