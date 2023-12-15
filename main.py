import cv2
import os
import face_recognition
import numpy as np

# Initialize the classifier
cascPath = os.path.dirname(cv2._file_) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Specify the path to the dataset folder
dataset_path = "D:\python\facerecognation\Face-Recognition-master\dataset"

# Create lists to store known faces and labels
known_faces = []
known_labels = []

# Loop through all files in the dataset folder
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(dataset_path, filename))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_img)[0]
        known_faces.append(face_encoding)
        label = os.path.splitext(filename)[0]  # get the name from the filename without extension
        known_labels.append(label)

# Open a connection to the webcam (usually 0 or 1, depending on your setup)
cap = cv2.VideoCapture(0)

while True:
    # Capture video frame-by-frame
    ret, img = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Iterate through detected faces in the frame
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_face)[0]
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        distances = face_recognition.face_distance(known_faces, face_encoding)

        # Find the index of the lowest distance
        best_match_index = np.argmin(distances)

        # If there is a match, display the name of the person on the frame, along with the rectangle around the face.
        # If there is no match, display a message like "Unknown" or "Name not found".
        if matches[best_match_index]:
            name = known_labels[best_match_index]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face recognition', img)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()