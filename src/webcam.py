import cv2

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

cap = cv2.VideoCapture(0)

# ---------- SMOOTHING VARIABLES ----------
last_emotion = "Normal"
stable_emotion = "Normal"
emotion_counter = 0
STABILITY_THRESHOLD = 6  # كلما زاد كلما ولى ثابت أكثر
confidence = 50
# ----------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=7
        )

        lower_face_gray = roi_gray[int(h/2):h, :]
        smiles = smile_cascade.detectMultiScale(
            lower_face_gray, scaleFactor=1.3, minNeighbors=5
        )

        # -------- RAW EMOTION DETECTION --------
        detected_emotion = "Normal"
        detected_confidence = 55

        if len(smiles) > 0:
            detected_emotion = "Happy"
            detected_confidence = min(90, 60 + len(smiles) * 10)

        elif len(eyes) == 0:
            detected_emotion = "Sad"
            detected_confidence = 65

        else:
            detected_emotion = "Normal"
            detected_confidence = 55 + min(len(eyes) * 5, 15)
        # --------------------------------------

        # -------- SMOOTHING LOGIC --------
        if detected_emotion == last_emotion:
            emotion_counter += 1
        else:
            emotion_counter = 0
            last_emotion = detected_emotion

        if emotion_counter >= STABILITY_THRESHOLD:
            stable_emotion = detected_emotion
            confidence = detected_confidence
        # --------------------------------

        # Colors
        if stable_emotion == "Happy":
            color = (0, 255, 0)      # Green
        elif stable_emotion == "Sad":
            color = (0, 0, 255)      # Red
        else:
            color = (0, 165, 255)    # Orange

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{stable_emotion} ({confidence}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Emotion Detection (Smoothed)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
