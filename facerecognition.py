import cv2
import pickle
import sqlite3
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load label map
with open("trainer/labels.pickle", "rb") as f:
    label_map = pickle.load(f)

# Connect database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Get today's date
today = datetime.now().strftime("%Y-%m-%d")

# To avoid duplicate attendance
marked_students = set()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))

        label, confidence = recognizer.predict(face)

        if confidence < 70:

            student_id = label_map[label]

            if student_id not in marked_students:

                cursor.execute(
                    "INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)",
                    (student_id, today, "Present")
                )

                conn.commit()
                marked_students.add(student_id)

            text = f"ID: {student_id}"

        else:
            text = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()