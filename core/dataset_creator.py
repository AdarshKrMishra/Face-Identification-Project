import cv2
import os
import sys
from config import DATASET_PATH, IMG_SIZE

# Get name from web
if len(sys.argv) > 1:
    person_name = sys.argv[1]
else:
    person_name = "Unknown"

path = f"{DATASET_PATH}/{person_name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
    "haarcascade_frontalface_default.xml")
)

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        cv2.imwrite(f"{path}/{count}.jpg", face)
        count += 1

        cv2.imshow("Captured Face", face)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Capture", frame)

    if cv2.waitKey(1)==27 or count>=200:
        break

cap.release()
cv2.destroyAllWindows()