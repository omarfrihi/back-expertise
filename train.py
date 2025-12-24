
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

from preprocessing.preprocessing import load_and_prepare_data
from pipeline.pipeline import build_pipeline
from evaluation.evaluation import evaluate_model
from visualization.visualization import (
    save_test_metrics,
    save_classification_report,
    save_cv_accuracy,
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
    save_top_words,
    save_learning_curve
)
from utils.utils import create_folders, save_model

# =========================================================
# 1. Setup
# =========================================================
create_folders()

# =========================================================
# 2. Load & prepare data
# =========================================================
df = load_and_prepare_data("data/train_news.csv")

# =========================================================
# 3. Train / Test split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

pd.DataFrame({"content": X_test, "label": y_test}).to_csv(
    "data/test_news.csv", index=False
)

# =========================================================
# 4. Build & train model
# =========================================================
pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

# =========================================================
# 5. Evaluation
# =========================================================
metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)

# =========================================================
# 6. Cross-validation
# =========================================================
cv_scores = cross_val_score(
    pipeline,
    df["content"],
    df["label"],
    cv=5,
    scoring="accuracy"
)

# =========================================================
# 7. Visualizations (ALL)
# =========================================================
save_test_metrics(
    metrics["accuracy"],
    metrics["precision"],
    metrics["recall"],
    metrics["f1"],
    "output/test_metrics.png"
)

save_classification_report(
    y_test,
    y_pred,
    "output/classification_report.png"
)

save_cv_accuracy(
    cv_scores,
    "output/cv_accuracy.png"
)

save_confusion_matrix(
    pipeline,
    X_test,
    y_test,
    "output/confusion_matrix.png"
)

save_roc_curve(
    y_test,
    y_proba,
    "output/roc_curve.png"
)

save_pr_curve(
    y_test,
    y_proba,
    "output/precision_recall_curve.png"
)

save_top_words(
    pipeline,
    "output/top_true_words.png",
    "output/top_false_words.png"
)

save_learning_curve(
    pipeline,
    df["content"],
    df["label"],
    "output/learning_curve.png"
)

# =========================================================
# 8. Save model
# =========================================================
save_model(pipeline, "model/fake_news_pipeline.joblib")
