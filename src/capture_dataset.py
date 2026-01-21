import cv2
import os

emotion = input("دخل emotion (happy / sad / normal): ")

save_path = f"../dataset/train/{emotion}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("اضغط SPACE باش تصوّر، Q باش تخرج")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture Dataset", gray)

    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE
        img_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(img_name, gray)
        print(f"📸 Image saved: {img_name}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Capture finished")
