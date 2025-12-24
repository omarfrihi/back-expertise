
import re
import pandas as pd

def clean_text(text: str) -> str:
    """Normalize and clean text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and prepare content column."""
    df = pd.read_csv(csv_path)
    df["content"] = (
        df.get("headline", "").fillna("") + " " +
        df.get("news", "").fillna("")
    ).str.strip()
    df = df[["content", "label"]].dropna()
    df["content"] = df["content"].apply(clean_text)
    return df
