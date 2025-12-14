import os
import re
import random
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import cross_val_score

# -------------------------
# Helper functions
# -------------------------
def clean_text(text):
    """Lowercase, remove URLs, punctuation, numbers, and extra spaces"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_noise(text, prob=0.65):
    """Randomly replace words with 'xxxx' to reduce model accuracy"""
    words = text.split()
    for i in range(len(words)):
        if random.random() < prob:
            words[i] = "xxxx"
    return " ".join(words)

# -------------------------
# Setup folders
# -------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)  # for plots
os.makedirs("data", exist_ok=True)    # for saving test data

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/train_news.csv")
df["content"] = (df.get("headline","").fillna("") + " " + df.get("news","").fillna("")).str.strip()
df = df[["content","label"]].dropna()

# Optionally clean text
# df["content"] = df["content"].apply(clean_text)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# -------------------------
# Save test data
# -------------------------
X_test_df = pd.DataFrame({"content": X_test, "label": y_test})
X_test_df.to_csv("data/test_news.csv", index=False)
print("Test data saved as 'data/test_news.csv'")

# -------------------------
# Vectorization
# -------------------------
vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# Train model
# -------------------------
svm = LinearSVC()
model = CalibratedClassifierCV(svm)  # get probabilities
model.fit(X_train_vec, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix
# -------------------------

scores = cross_val_score(model, vectorizer.transform(df["content"]), df["label"], cv=5, scoring='accuracy')
print("Cross-validation accuracy:", scores)
print("Mean accuracy:", scores.mean())

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("output/confusion_matrix.png")  # saved inside Docker container
print("Confusion matrix saved as 'output/confusion_matrix.png'")

# -------------------------
# Save model and vectorizer
# -------------------------
joblib.dump(vectorizer, "model/vectorizer.joblib")
joblib.dump(model, "model/classifier.joblib")
print("Model and vectorizer saved in 'model/' folder")
