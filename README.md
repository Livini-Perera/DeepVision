**DeepVision - Vehicle Detection and Classification System 🚗**

DeepVision: YOLOv8-based real-time vehicle detection system built as a university project. The system can detect and classify 8 types of vehicles in images with bounding boxes.
---

## 📋 Project Overview

**Goal:** Build an accurate vehicle detection and classification system using deep learning

**Vehicle Classes Detected:**
- 🚲 Bicycle
- 🚌 Bus
- 🚗 Car
- 🏍️ Motorcycle
- 🛺 Three-wheeler
- 🚜 Tractor
- 🚚 Truck
- 🚐 Van

---

## 👥 Team Structure

**6-Member Team (Working as groups):**

**Backend Development (ML & Data):**
- Pruthivi
- Adhil
- Budara

**Frontend Development (Application & UI):**
- Luke
- Victor
- Prince

---

## 🛠️ Tech Stack

**Machine Learning:**
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Google Colab (GPU training)

**Development:**
- Python 3.10 or above
- Jupyter Notebook
- Streamlit (for web app)
- Git/GitHub

**Dataset:**
- Roboflow

## 📊 Dataset

[Link for the Dataset](https://drive.google.com/drive/folders/1wkjz3ii1RQopnvucXk7H09KLDDxvaW-H?usp=drive_link)

Cannot upload the dataset due to size constraints

### Dataset Source

Our dataset was obtained from **Roboflow Universe** in YOLO format.

- **Total Images:** 4433 images
- **Classes:** 8 vehicle types
- **Format:** YOLO v8

### Dataset Structure

```
datasets/
└── Vehicle data set v5.v1i.yolov8/
    ├── train/
    │   ├── images/        # Training images
    │   └── labels/        # Training annotations
    ├── valid/
    │   ├── images/        # Validation images
    │   └── labels/        # Validation annotations
    ├── test/
    │   ├── images/        # Test images
    │   └── labels/        # Test annotations
    ├── data.yaml          # Dataset configuration
    ├── README.dataset     # Dataset information
    └── README.roboflow    # Roboflow attribution
```
