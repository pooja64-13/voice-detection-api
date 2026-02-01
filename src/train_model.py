import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from src.features import extract_basic_features



DATA_DIR = "data"


def load_dataset():
    X = []
    y = []

    # Human = 0
    human_dir = os.path.join(DATA_DIR, "human")
    for file in os.listdir(human_dir):
        path = os.path.join(human_dir, file)
        features = extract_basic_features(path)
        X.append(features)
        y.append(0)

    # AI = 1
    ai_dir = os.path.join(DATA_DIR, "ai")
    for file in os.listdir(ai_dir):
        path = os.path.join(ai_dir, file)
        features = extract_basic_features(path)
        X.append(features)
        y.append(1)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset()

    print("Class balance:", np.bincount(y))

    # Train on full dataset (no split – hackathon safe baseline)
    X_train, y_train = X, y

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "voice_detector.pkl")
    print("Model saved as voice_detector.pkl")
