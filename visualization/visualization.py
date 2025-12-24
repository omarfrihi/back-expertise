
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

def save_test_metrics(acc, precision, recall, f1, path):
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [acc, precision, recall, f1]
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("Test Metrics")
    plt.savefig(path)
    plt.close()

def save_classification_report(y_test, y_pred, path):
    report = classification_report(
    y_test,
    y_pred,
    digits=4,
    output_dict=True
)

    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.table(
    cellText=report_df.round(3).values,
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc="center"
)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()






def save_cv_accuracy(cv_scores, path):
    plt.bar(range(1, len(cv_scores) + 1), cv_scores)
    plt.axhline(y=cv_scores.mean(), linestyle="--")
    plt.title("Cross-Validation Accuracy")
    plt.savefig(path)
    plt.close()

def save_confusion_matrix(model, X_test, y_test, path):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.savefig(path)
    plt.close()

def save_roc_curve(y_test, y_proba, path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr)
    plt.savefig(path)
    plt.close()

def save_pr_curve(y_test, y_proba, path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision)
    plt.savefig(path)
    plt.close()

def save_top_words(pipeline, path_true, path_false, top_n=15):
    tfidf = pipeline.named_steps["tfidf"]
    calibrated = pipeline.named_steps["clf"]
    clf = calibrated.calibrated_classifiers_[0].estimator
    features = tfidf.get_feature_names_out()
    coef = clf.coef_[0]

    idx_true = np.argsort(coef)[-top_n:]
    idx_false = np.argsort(coef)[:top_n]

    plt.barh(features[idx_true], coef[idx_true])
    plt.savefig(path_true)
    plt.close()

    plt.barh(features[idx_false], coef[idx_false])
    plt.savefig(path_false)
    plt.close()

def save_learning_curve(pipeline, X, y, path):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=3, scoring="accuracy", n_jobs=1,
        train_sizes=[0.2, 0.5, 1.0]
    )
    plt.plot(train_sizes, train_scores.mean(axis=1))
    plt.plot(train_sizes, val_scores.mean(axis=1))
    plt.savefig(path)
    plt.close()
