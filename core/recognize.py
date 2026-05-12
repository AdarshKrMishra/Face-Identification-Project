import cv2
import numpy as np
import pickle
import os

from config import MODEL_PATH, IMG_SIZE, THRESHOLD
from lbp import compute_lbp, compute_lbph

with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

print("Model Loaded")

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
    "haarcascade_frontalface_default.xml")
)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = cv2.resize(gray[y:y+h,x:x+w],(IMG_SIZE,IMG_SIZE))

        lbp = compute_lbp(face)
        features = compute_lbph(lbp)

        probs = model.predict_proba([features])[0]
        idx = np.argmax(probs)
        name = model.classes_[idx]
        conf = probs[idx]*100

        label = name if conf>THRESHOLD else "Unknown"

        cv2.putText(frame,f"{label} ({conf:.1f}%)",
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()