import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="DeepVision - Vehicle Detection",
    layout="wide"
)

# -------------------------------------------------
# LOAD CSS
# -------------------------------------------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your trained model file

model = load_model()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    # Logo (must be in same folder)
    st.image("logo.png", width=200)

    st.markdown("## Configuration")

    input_mode = st.radio(
        "Input Mode:",
        ["Single Image", "Multiple Images"]
    )

    confidence = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.5
    )

    st.markdown("---")
    st.markdown("### Detected Classes")

    class_emojis = {
        "car": "🚗",
        "van":"🚐",
        "truck": "🚛",
        "bus": "🚌",
        "motorbike": "🏍️",
        "bicycle": "🚲",
        "three wheeler/ tuk tuk": "🛺"
    }

    for cls, emoji in class_emojis.items():
        st.markdown(f"{emoji} {cls.title()}")

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Vehicle Detection System</h1>
    <p>Advanced YOLOv8-based Vehicle Recognition and Classification</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FEATURES SECTION
# -------------------------------------------------
st.markdown("""
<div class="card">
    <h2>Features:</h2>
    <ul>
        <li>Real-time Vehicle Detection using YOLOv8</li>
        <li>Vehicle Classification (Car, Bus, Truck, Motorbike, Bicycle)</li>
        <li>Single and Multiple Image Processing</li>
        <li>Adjustable Confidence Threshold</li>
        <li>Per-Class Vehicle Count</li>
        <li>Average Confidence Calculation</li>
        <li>Automated Structured Report Generation (CSV)</li>
        <li>Professional Dashboard Interface</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# GET STARTED SECTION
# -------------------------------------------------
st.markdown("""
<div class="card">
    <h2>Get Started:</h2>
    <ol>
        <li>Select the input mode from the sidebar.</li>
        <li>Adjust the confidence threshold if required.</li>
        <li>Upload your image(s).</li>
        <li>Wait for the detection process to complete.</li>
        <li>Review the structured detection report.</li>
        <li>Download the final CSV report.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# IMAGE PROCESSING FUNCTION
# -------------------------------------------------
def process_image(image, filename):
    results = model(image, conf=confidence)
    result = results[0]

    boxes = result.boxes
    names = model.names

    class_data = {}

    for box in boxes:
        cls_id = int(box.cls)
        cls_name = names[cls_id]
        conf_score = float(box.conf)

        if cls_name not in class_data:
            class_data[cls_name] = {
                "count": 0,
                "confidences": []
            }

        class_data[cls_name]["count"] += 1
        class_data[cls_name]["confidences"].append(conf_score)

    report_rows = []

    for cls_name, values in class_data.items():
        avg_conf = sum(values["confidences"]) / len(values["confidences"])

        report_rows.append({
            "Image Name": filename,
            "Vehicle Class": cls_name,
            "Count": values["count"],
            "Average Confidence": round(avg_conf, 3)
        })

    return result.plot(), report_rows

# -------------------------------------------------
# SINGLE IMAGE MODE
# -------------------------------------------------
if input_mode == "Single Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        output_image, report_rows = process_image(image, uploaded_file.name)

        with col2:
            st.image(output_image, caption="Detected Image", use_column_width=True)

        df_report = pd.DataFrame(report_rows)

        st.subheader("Final Detection Report")
        st.dataframe(df_report, hide_index=True)

        csv = df_report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Final Report",
            csv,
            "final_vehicle_detection_report.csv",
            "text/csv"
        )

# -------------------------------------------------
# MULTIPLE IMAGE MODE
# -------------------------------------------------
elif input_mode == "Multiple Images":

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        all_reports = []

        for file in uploaded_files:
            image = Image.open(file)
            _, report_rows = process_image(image, file.name)
            all_reports.extend(report_rows)

        df_report = pd.DataFrame(all_reports)

        st.subheader("Batch Final Detection Report")
        st.dataframe(df_report, hide_index=True)

        csv = df_report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Batch Final Report",
            csv,
            "batch_final_vehicle_detection_report.csv",
            "text/csv"
        )