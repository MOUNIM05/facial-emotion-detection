# 🎭 Real-Time Emotion Detection with OpenCV

A **real-time facial emotion detection system** built using **Python** and **OpenCV**.  
This project analyzes live webcam video to detect faces and classify basic emotions using computer vision techniques.

Designed as a **portfolio-ready project** for students and junior developers interested in **AI**, **Computer Vision**, and **real-time systems**.

---

## 🔎 Overview

The application performs the following tasks:

- Captures live video from a webcam
- Detects human faces in real time
- Analyzes facial features (eyes and smile)
- Classifies emotions into:
  - 😄 Happy
  - 😐 Normal
  - 😢 Sad
- Displays:
  - Emotion label
  - Color-coded bounding box
  - Confidence percentage

---

## 🎯 Why This Project Is Relevant

This project demonstrates:

- Practical use of **OpenCV** for real-time video processing
- Understanding of **face detection pipelines**
- Emotion classification using **heuristic-based logic**
- Clean Python project structure
- Readiness for **future deep learning integration**
- A strong foundation for **AI / Computer Vision roles**

---

## 🛠️ Technologies Used

- Python 3
- OpenCV
- NumPy
- Haar Cascade Classifiers
- Webcam (real-time stream)

---

## 🎨 Emotion Classification & Colors

| Emotion | Detection Logic | Display Color |
|-------|----------------|---------------|
| 😄 Happy | Smile detected | 🟢 Green |
| 😐 Normal | Eyes detected without smile | 🟠 Orange |
| 😢 Sad | Face detected without smile or clear eyes | 🔴 Red |

---

## 📁 Project Structure

emotion-detection-opencv/
│
├── dataset/
│ ├── train/
│ │ ├── happy/
│ │ ├── normal/
│ │ └── sad/
│ └── test/
│ ├── happy/
│ ├── normal/
│ └── sad/
│
├── model/
│ └── emotion_model.h5
│
├── src/
│ ├── face_utils.py # Face, eye & smile detection helpers
│ ├── capture_data.py # Dataset capture tool
│ ├── train.py # Training logic (future CNN use)
│ └── webcam.py # Main real-time emotion detection script
│
├── assets/
│ ├── demo.gif
│ ├── happy.png
│ ├── normal.png
│ └── sad.png
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ Installation & Setup
### 1️⃣ Clone the repository
git clone https://github.com/USERNAME/emotion-detection-opencv.git
cd emotion-detection-opencv
---
##2️⃣ Create a virtual environment (recommended)
bash
Copy code
python -m venv venv
source venv/Scripts/activate
---
##3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
---
##▶️ Run the Application
Make sure that:
Your webcam is connected
No other application is using the camera
python src/webcam.py
---
##🧠 Emotion Detection Logic
This version uses a heuristic-based computer vision approach:
Happy → Smile detected
Normal → Eyes detected without smile
Sad → Face detected without smile or clear eyes
A confidence percentage (%) is calculated based on:
Face detection quality
Eye detection reliability
Smile detection consistency
---
##📊 Example Output
makefile
Copy code
Emotion: Happy 😄
Confidence: 81%
---
##🧪 Use Cases
OpenCV practice projects
Computer Vision learning
AI student portfolios
Real-time webcam experiments
GitHub technical showcase
---
##🚀 Future Improvements
CNN-based emotion classifier (Deep Learning)
Facial landmarks detection
Emotion smoothing over time
Dataset auto-training pipeline
GUI interface (Tkinter / PyQt)
Screenshot & logging features
---
##👤 Author
Abdelmounim Maani
AI & Computer Vision Student
Junior Python / OpenCV Developer
Open to internships and junior AI / CV opportunities.
---
##📜 License
This project is licensed under the MIT License.

