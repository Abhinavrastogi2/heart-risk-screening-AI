# 🤖 AI Vision & Object Detection Dashboard

An interactive, production-grade web application built with **Streamlit**, **YOLOv8** (Object Detection), and **Salesforce BLIP** (Image Captioning).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![BLIP](https://img.shields.io/badge/BLIP-HuggingFace-orange)

---

## 🌟 Key Features

- **🔍 YOLOv8 Real-Time Object Detection**: Detect multiple objects with bounding boxes, confidence scores, and bounding box coordinates.
- **📝 BLIP Natural Language Captioning**: Generate detailed AI captions describing what is inside any photo.
- **📷 Multiple Input Sources**: Support for file uploads, workspace sample images, and direct webcam capture.
- **🎛️ Live Threshold & Filter Controls**: Filter detected objects by confidence score and specific class types in real time.
- **📊 Interactive Analytics**: Detailed statistical summaries, class count bar charts, and scatter plots.
- **✂️ Object Crop Gallery**: Automatically extract and view individual crops of detected objects.
- **💾 CSV Export & History**: Save detection records to `detection_log.csv` and download logs directly.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser!

---

## 📁 Repository Structure

```text
├── app.py                 # Streamlit web dashboard source code
├── check.ipynb            # Original Jupyter Notebook pipeline
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
├── run_dashboard.bat      # 1-click Windows launcher
└── detection_log.csv      # Log file for object detections
```

---
## 🤝 License
Distributed under the MIT License.
By Abhinav Rastogi
