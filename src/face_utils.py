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


def detect_faces(gray):
    return face_cascade.detectMultiScale(gray, 1.3, 5)


def detect_eyes(gray_face):
    return eye_cascade.detectMultiScale(gray_face, 1.1, 5)


def detect_smile(gray_face):
    return smile_cascade.detectMultiScale(gray_face, 1.7, 20)
