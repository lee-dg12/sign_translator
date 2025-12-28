import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

X = np.load("X.npy")
y = np.load("y.npy")

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(C=10, kernel="rbf", gamma="scale", probability=True)),
])

model.fit(Xtr, ytr)
pred = model.predict(Xte)

print(classification_report(yte, pred))

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/hand_svm.joblib")
print("saved: artifacts/hand_svm.joblib")
