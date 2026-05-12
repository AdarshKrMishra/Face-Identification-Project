import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from config import DATASET_PATH, MODEL_PATH, IMG_SIZE
from lbp import compute_lbp, compute_lbph

X, y = [], []

for person in os.listdir(DATASET_PATH):
    for img_name in os.listdir(f"{DATASET_PATH}/{person}"):

        img = cv2.imread(f"{DATASET_PATH}/{person}/{img_name}",0)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        lbp = compute_lbp(img)
        features = compute_lbph(lbp)

        X.append(features)
        y.append(person)

X, y = np.array(X), np.array(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))

cm = confusion_matrix(y_test,pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

with open(MODEL_PATH,"wb") as f:
    pickle.dump(model,f)

print("Model Saved")