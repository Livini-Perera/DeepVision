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

## Frontend

## 📁 utils/ Package

**Purpose:** Contains helper modules that power the VehicleDetectionSystem

**Structure:**
```
utils/
├── __init__.py        # Makes utils a Python package
├── detector.py        # Vehicle detection logic (YOLOv8)
├── reporter.py        # Report and statistics generation
└── image_helper.py    # Logo and image utilities
```

**What Each Module Does:**
- `detector.py` - Core vehicle detection using YOLOv8 model
- `reporter.py` - Generates CSV reports and summary statistics
- `image_helper.py` - Handles logo display for the UI
- `__init__.py` - Makes the folder a package and controls imports

---

## 📦 __init__.py

**Purpose:** Makes `utils/` a Python package and enables clean imports

**What It Does:**

- Converts the `utils/` folder into an importable Python package  
- Exports `VehicleDetector` and `generate_report` for easy access  
- Enables shorthand imports: `from utils import VehicleDetector` instead of `from utils.detector import VehicleDetector`

**Exports:**
- `VehicleDetector` - Main detection class
- `generate_report` - CSV report generation function

**The `__all__` List:**  
Defines what's publicly available when someone imports from the package

**Why It's Needed:**  
Without this file, Python treats `utils/` as a regular folder, not a package, and imports won't work.

## 📊detector.py

**Purpose:** Core vehicle detection using YOLOv8

**Main Class:**
- `VehicleDetector` - Handles all detection operations

**Key Methods:**
- `detect(image_path)` - Detect vehicles in single image
- `annotate_image(image_path, detections, output_path)` - Draw bounding boxes
- `process_single_image(image_path, output_dir)` - Complete pipeline for one image
- `process_folder(folder_path, output_dir)` - Batch processing

## 📊 reporter.py

**Purpose:** Generate reports and statistics from detection results

**Key Functions:**

- `generate_report(results, output_path)` - Creates detailed CSV report from batch detection results  
  - Parameters: Detection results list, output file path  
  - Returns: pandas DataFrame  
  - Output: CSV file with per-image breakdown (image name, vehicle counts, dominant class, confidence scores)

- `generate_summary_stats(results)` - Calculates overall statistics across all processed images  
  - Parameters: Detection results list  
  - Returns: Dictionary with total_images, total_vehicles, avg_confidence, class_distribution  
  - Use for: Dashboard metrics, overall analysis, batch processing summaries

**Dependencies:**
- `pandas` - Data manipulation and CSV generation

**CSV Report Columns:**
- Image Name - Filename of processed image
- Total Vehicles - Number of vehicles detected
- Dominant Class - Most common vehicle type in image
- Average Confidence - Mean detection confidence (percentage)
- [Class] Count - Individual counts for each vehicle class (car, truck, bus, etc.)

**Summary Statistics Output:**
- `total_images` - Number of images processed
- `total_vehicles` - Total vehicles detected across all images
- `avg_confidence` - Average confidence score (0-1)
- `class_distribution` - Dictionary with count per vehicle class

**Integration:** Works seamlessly with `detector.py` output for batch processing workflows
