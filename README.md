# 😷 Face Mask Detection System

A **real-time face mask detection application** using **computer vision** and **deep learning**. 
Supports **live webcam streaming** and **image uploads** for monitoring mask compliance in public areas, workplaces, or security checkpoints.

---

## 📝 About the Project
The Face Mask Detection System identifies whether individuals are wearing a mask in real-time. 
It uses Python, OpenCV, and TensorFlow/Keras to process video or image input, and Streamlit for a user-friendly web interface.

---

## 🚀 Features
- 📷 **Real-time detection** via webcam  
- 🖼️ **Image upload** support for testing  
- 🎯 **High accuracy** in varied lighting conditions  
- 💻 **Web-based interface** powered by Streamlit  
- 🌍 **Cross-platform** support (Windows, macOS, Linux)  

---

## 🛠️ Tech Stack
- **Python** – Core programming language  
- **OpenCV** – Face detection & image processing  
- **TensorFlow / Keras** – Deep learning model  
- **Streamlit + Streamlit-WebRTC** – Interactive web app  
- **NumPy, Pillow, scikit-learn, Matplotlib, imutils** – Supporting libraries  

---

## 📋 Requirements
- Python **3.7 or higher**  
- **pip** package manager  
- A **webcam** (for real-time detection)  

---

## ⚙️ Installation

### 1. Clone the repository
```
git clone https://github.com/Sudarsan003-max/FACE-MASK-DETECTOR.git
cd FACE-MASK-DETECTOR
2. Create a virtual environment (recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
🖥️ Usage
Run the Streamlit app:


Copy code
streamlit run app.py
Select Webcam Mode or Image Upload
```
---

🏗️ Project Structure
```
FACE-MASK-DETECTOR/
├── app.py                 # Main Streamlit application
├── detect_mask_image.py   # Image detection pipeline
├── detect_mask_video.py   # Video/webcam detection pipeline
├── convert_to_onnx.py     # Model conversion script
├── face_detector/         # Pre-trained face detection models
├── mask_detector.model    # Trained mask detection model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```
---
# 🤝 Contributing

Contributions are welcome!

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m "Add AmazingFeature")

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

