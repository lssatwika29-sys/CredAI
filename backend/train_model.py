
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# 1. Load dataset
df = pd.read_csv("data/Loan_default.csv")
df = df.drop_duplicates().copy()

X = df.drop(columns=["LoanID", "Default"])
y = df["Default"]

print("Dataset shape:", df.shape)
print("\nDefault distribution:")
print(y.value_counts())

# 2. Split into training, validation and testing data
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# 3. Identify columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# 5. Build improved Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt"
    ))
])

# 6. Train model
print("\nTraining optimized model. Please wait...")
model.fit(X_train, y_train)

# 7. Find best threshold using validation data
val_probabilities = model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

best_threshold = 0.5
best_f2 = -1

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = fbeta_score(
        y_val,
        val_predictions,
        beta=2,
        zero_division=0
    )

    if score > best_f2:
        best_f2 = score
        best_threshold = threshold

print("\nSelected validation threshold:", round(float(best_threshold), 4))
print("Validation F2 score:", round(float(best_f2), 4))

# 8. Get test probabilities
test_probabilities = model.predict_proba(X_test)[:, 1]

# 9. Compare different thresholds
thresholds_to_test = [
    0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50
]

comparison_results = []

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in thresholds_to_test:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_test,
        predictions,
        beta=2,
        zero_division=0
    )

    comparison_results.append({
        "Threshold": threshold,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "F2 Score": round(f2, 4)
    })

    print(
        f"\nThreshold: {threshold:.2f}"
        f"\nAccuracy: {accuracy:.4f}"
        f"\nPrecision: {precision:.4f}"
        f"\nRecall: {recall:.4f}"
        f"\nF1 Score: {f1:.4f}"
        f"\nF2 Score: {f2:.4f}"
    )

# 10. Display comparison table
comparison_df = pd.DataFrame(comparison_results)

print("\n========== FINAL COMPARISON TABLE ==========")
print(comparison_df.to_string(index=False))

# 11. Save comparison results
comparison_df.to_csv(
    "threshold_comparison.csv",
    index=False
)

print("\nComparison saved as threshold_comparison.csv")

# 12. Evaluate selected validation threshold on test data
final_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n========== SELECTED THRESHOLD TEST RESULTS ==========")

print("Threshold:", round(float(best_threshold), 4))

print(
    "Accuracy:",
    round(accuracy_score(y_test, final_predictions), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, final_predictions, zero_division=0), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, final_predictions, zero_division=0), 4)
)

print(
    "F1 Score:",
    round(f1_score(y_test, final_predictions, zero_division=0), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    final_predictions,
    zero_division=0
))

# 13. Save optimized model separately
joblib.dump(
    model,
    "loan_default_model_optimized.joblib"
)

# 14. Save selected threshold
with open("model_threshold.json", "w") as file:
    json.dump(
        {"threshold": float(best_threshold)},
        file,
        indent=4
    )

print("\n========== FILES SAVED SUCCESSFULLY ==========")
print("Optimized model: loan_default_model_optimized.joblib")
print("Selected threshold: model_threshold.json")
print("Threshold comparison: threshold_comparison.csv")
