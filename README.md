<header style="width: 100%; height: 10px;">
  <img src="DeepVision Header.gif" alt="Header Image" style="width: 100%; height: 20%;" />
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

---

## 🎨 Frontend Application

### **Overview**

A professional Streamlit-based web application that provides an intuitive interface for vehicle detection. The frontend allows users to upload images, view detection results with bounding boxes, and download structured CSV reports.

### **Features**

* ✅ **Real-time Vehicle Detection** - Upload and process images instantly
* ✅ **Single & Batch Processing** - Process one or multiple images at once
* ✅ **Adjustable Confidence Threshold** - Fine-tune detection sensitivity (0.1 - 1.0)
* ✅ **Visual Detection Results** - See bounding boxes and labels on images
* ✅ **Per-Class Vehicle Count** - Automatic counting by vehicle type
* ✅ **Average Confidence Calculation** - Get confidence scores for each detection
* ✅ **CSV Report Generation** - Download structured reports for analysis
* ✅ **Professional Dashboard Interface** - Clean, modern UI with custom styling
* ✅ **Responsive Layout** - Works on different screen sizes

### **Application Structure**
```
VehicleDetectionSystem/
├── deepvision_app.py        # Main Streamlit application
├── styles.css                # Custom CSS styling
├── best.pt                   # Trained YOLOv8 model
├── logo.png                  # DeepVision logo
├── requirements.txt          # Python dependencies
└── README.md                 # Application documentation
```

### **User Interface Components**

#### **Sidebar**
* Project title and description
* Input mode selection (Single Image / Multiple Images)
* Confidence threshold slider
* List of detected vehicle classes with emojis

#### **Main Area**
* Hero section with project title
* Features list
* Usage instructions
* Image upload area
* Side-by-side comparison (Original vs Detected)
* Detection results table
* CSV report download button

### **Output Format**

The application generates CSV reports with the following structure:

| Column | Description |
|--------|-------------|
| **Image Name** | Name of the processed image |
| **Vehicle Class** | Type of vehicle detected |
| **Count** | Number of vehicles of this class |
| **Average Confidence** | Mean confidence score (0-1) |

**Example CSV Output:**
```csv
Image Name,Vehicle Class,Count,Average Confidence
traffic.jpg,car,12,0.876
traffic.jpg,bus,3,0.912
traffic.jpg,motorcycle,5,0.834
```

---

## 🚀 Installation & Setup

### **Prerequisites**

* Python 3.10 or higher
* pip (Python package manager)
* Git

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/deepvision.git
cd deepvision
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Download the Model**

Download `best.pt` (trained model, 22MB) from [Google Drive](#) and place it in the `VehicleDetectionSystem/` directory.

**Note:** The model file is not included in the repository due to its size.

### **Step 4: Run the Application**
```bash
cd VehicleDetectionSystem
streamlit run deepvision_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 📦 Dependencies
```txt
streamlit==1.28.0
ultralytics==8.0.196
pillow==10.0.1
pandas==2.1.1
opencv-python==4.8.1.78
torch>=2.0.0
```

Install all dependencies at once:
```bash
pip install -r requirements.txt
```

---

## 📖 How to Use the Application

### **Single Image Detection**

1. **Select Input Mode:** Choose "Single Image" from the sidebar
2. **Adjust Confidence:** Set the confidence threshold (default: 0.25)
3. **Upload Image:** Click "Upload an image" and select a JPG/PNG file
4. **View Results:**
   - Original image displayed on the left
   - Detected image with bounding boxes on the right
   - Detection report table below with vehicle counts
5. **Download Report:** Click "Download Report (CSV)" to save results

### **Batch Image Detection**

1. **Select Input Mode:** Choose "Multiple Images" from the sidebar
2. **Adjust Confidence:** Set the confidence threshold
3. **Upload Images:** Select multiple JPG/PNG files at once
4. **View Results:**
   - Combined detection report for all images
   - Organized by image name and vehicle class
5. **Download Report:** Click "Download Batch Report (CSV)"

---

## 🔧 Configuration

### **Confidence Threshold**

Adjust the confidence threshold to control detection sensitivity:

* **Lower (0.1 - 0.3):** More detections, may include false positives
* **Default (0.25):** Balanced detection, matches training configuration
* **Higher (0.5 - 1.0):** Fewer detections, only high-confidence predictions

### **Model Path**

If your model is in a different location, update line 28 in `deepvision_app.py`:
```python
@st.cache_resource
def load_model():
    return YOLO("path/to/your/best.pt")
```

---

## 🐛 Troubleshooting

### **Model File Not Found**
```
Error: [Errno 2] No such file or directory: 'best.pt'
```
**Solution:** Download `best.pt` and place it in the `VehicleDetectionSystem/` directory

### **Import Errors**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution:** Install dependencies using `pip install -r requirements.txt`

### **PyTorch Compatibility Error**
```
UnpicklingError: Weights only load failed
```
**Solution:** The fix is already included in the training notebook. Make sure to run the PyTorch patch cell before loading the model.

### **No Detections**
If no vehicles are detected in your image:
* Lower the confidence threshold (try 0.15)
* Ensure the image contains one of the 8 vehicle classes
* Check image quality (not too blurry or dark)

### **CSS Not Loading**
If the interface looks unstyled:
* Ensure `styles.css` is in the same directory as `deepvision_app.py`
* Refresh the browser (Ctrl+F5)
* Check browser console for errors

---

## 🔮 Future Enhancements

Potential improvements for future versions:

* [ ] **Video Processing** - Real-time vehicle detection in video files
* [ ] **Live Camera Feed** - Detect vehicles from webcam
* [ ] **Advanced Analytics** - Vehicle tracking, speed estimation
* [ ] **Database Integration** - Store detection history
* [ ] **REST API** - API endpoint for programmatic access
* [ ] **Mobile App** - Native mobile application
* [ ] **Model Optimization** - Faster inference with TensorRT/ONNX
* [ ] **More Vehicle Classes** - Expand to 15+ vehicle types
* [ ] **Cloud Deployment** - Deploy on AWS/GCP/Azure

---

## 📄 License

This project is part of an academic assignment for University of SOuth Wales.

For educational purposes only.

---

## 🙏 Acknowledgments

* **Ultralytics** - For the YOLOv8 framework
* **Streamlit** - For the web application framework
* **Roboflow** - For dataset hosting and management
* **Google Colab** - For free GPU training resources
* Our university professors for guidance and support

---

## 📚 References

1. **YOLOv8 Documentation:** [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
2. **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
3. **Dataset Source:** Roboflow Universe - Vehicle Detection Dataset
4. **PyTorch Documentation:** [https://pytorch.org/docs/](https://pytorch.org/docs/)
5. **OpenCV Documentation:** [https://docs.opencv.org/](https://docs.opencv.org/)

---

## ⚙️ System Requirements

### **For Training:**
* GPU: NVIDIA GPU with CUDA support (or use Google Colab)
* RAM: 12GB+ recommended
* Storage: 5GB free space
* Internet: Required for Colab

### **For Running the Application:**
* CPU: Dual-core processor (minimum)
* RAM: 4GB (8GB recommended)
* Storage: 1GB free space
* OS: Windows 10/11, macOS 10.14+, or Linux

---

**Made with ❤️ by the DeepVision Team**
