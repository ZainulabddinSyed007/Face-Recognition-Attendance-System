import cv2
import os
import sqlite3

# Connect to database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Ask student details
student_id = input("Enter Student ID: ")
name = input("Enter Student Name: ")
department = input("Enter Department: ")
year = input("Enter Year: ")
email = input("Enter Email: ")

# Insert student details into database
cursor.execute(
    "INSERT INTO students (student_id, name, department, year, email) VALUES (?, ?, ?, ?, ?)",
    (student_id, name, department, year, email)
)

conn.commit()
conn.close()

# Create dataset folder
folder_name = f"{student_id}_{name}"
path = os.path.join("dataset", folder_name)
os.makedirs(path, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0

print("Look at the camera. Capturing faces...")

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
        count += 1

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize face
        face = cv2.resize(face, (200, 200))

        # Save image
        file_path = os.path.join(path, f"{count}.jpg")
        cv2.imwrite(file_path, face)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop after 50 images
    if count >= 50:
        break

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Face capture completed!")