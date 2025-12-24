
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def build_pipeline() -> Pipeline:
    """Build ML pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"),
            method="sigmoid",
            cv=5
        ))
    ])
