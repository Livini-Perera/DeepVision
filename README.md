<header style="width: 100%; height: 10px;">
  <img src="header.gif" alt="Header Image" style="width: 100%; height: 20%;" />
</header>


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

## 🎉 Model Training

### Model Performance

Successfully trained a YOLOv8s model on our vehicle detection dataset

**Final Results:**
- **Model:** YOLOv8s
- **Dataset:** 4423 images
- **Training Time:** 2-2.5 hours (Google Colab T4 GPU)
- **Final Metrics:**
  - **mAP:** [82.74%]
  - **Precision:** [84.28%]
  - **Recall:** [77.48%]

### Per-Class Performance

| Vehicle Type | Precision | Recall | mAP     |
|--------------|-----------|--------|---------|
| Bicycle      | 56.5%     | 54.5%  | 53.7%   |
| Bus          | 85.6%     | 86%    | 89.7%   |
| Car          | 80.6%     | 71.1%  | 79%     |
| Motorcycle   | 88.9%     | 74.5%  | 83.1%   |
| Three-wheeler| 90.4%     | 86.4%  | 93.2%   |
| Tractor      | 92.2%     | 84.1%  | 85%     |
| Truck        | 86.7%     | 77.1%  | 84.3%   |
| Van          | 93.4%     | 86%    | 94%     |

### Training Environment

- **Platform:** Google Colab
- **GPU:** Tesla T4
- **Framework:** YOLOv8 (Ultralytics)
- **PyTorch Version:** 2.9.0 (Default)
  
  Error: PyTorch 2.6+ changed weights_only default from False to True, blocking YOLOv8 model loading with UnpicklingError because YOLO classes weren't in the safe_globals whitelist.
  
   Fix: Patched torch.load to use weights_only=False for trusted Ultralytics weights.
