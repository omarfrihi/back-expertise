
import os, joblib

def create_folders():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)

def save_model(model, path):
    joblib.dump(model, path)
