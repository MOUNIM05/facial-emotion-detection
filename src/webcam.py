import cv2
import numpy as np
from face_utils import detect_faces, detect_eyes, detect_smile

# Webcam
cap = cv2.VideoCapture(0)

print("🎥 Webcam started — Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        eyes = detect_eyes(face_gray)
        smiles = detect_smile(face_gray)

        # =====================
        # EMOTION LOGIC
        # =====================
        if len(smiles) > 0:
            emotion = "Happy 😄"
            color = (0, 255, 0)       # Green
            confidence = min(95, 70 + len(smiles) * 10)

        elif len(eyes) >= 2:
            emotion = "Normal 😐"
            color = (0, 165, 255)     # Orange
            confidence = 60 + len(eyes) * 10

        else:
            emotion = "Sad 😢"
            color = (0, 0, 255)       # Red
            confidence = 65

        confidence = min(confidence, 100)

        # =====================
        # DRAW RESULTS
        # =====================
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        text = f"{emotion}  {confidence}%"
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
