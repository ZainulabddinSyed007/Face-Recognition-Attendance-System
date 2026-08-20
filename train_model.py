import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"

faces = []
labels = []
label_map = {}

current_label = 0

for folder_name in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, folder_name)

    if os.path.isdir(student_path):

        # folder format: StudentID_Name
        student_id = folder_name.split("_")[0]

        label_map[current_label] = student_id

        for image_name in os.listdir(student_path):
            image_path = os.path.join(student_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

os.makedirs("trainer", exist_ok=True)

recognizer.save("trainer/trainer.yml")

# Save label map
with open("trainer/labels.pickle", "wb") as f:
    pickle.dump(label_map, f)

print("Model trained successfully!")