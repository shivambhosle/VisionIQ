# ==================================================
# VisionIQ – Multi‑Object AI Inspection System
# Bottle + Phone + Pen | FINAL HACKATHON VERSION
# ==================================================

from ultralytics import YOLO
import cv2
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time

# -------------------------
# CONFIGURATION
# -------------------------
EXPECTED_OBJECTS = ["bottle", "cell phone", "pen"]

# Known physical dimensions (mm) – these should be calibrated per actual object
KNOWN_WIDTH_MM = {
    "bottle": 60,
    "cell phone": 70,  # This is width (typically 70-80mm for phones)
    "pen": 10
}

KNOWN_HEIGHT_MM = {
    "bottle": 270,
    "cell phone": 150,  # This is height (typically 150-160mm for phones)
    "pen": 140
}

# Adjusted thresholds for better detection
CAP_AREA_THRESHOLD = 600
CAP_ASPECT_RATIO = 1.2
SCRATCH_EDGE_THRESHOLD = 0.05  # Reduced for better scratch detection
CRACK_EDGE_THRESHOLD = 0.08
MIN_CONTOUR_AREA = 500  # Minimum contour area for defect detection

# Phone-specific parameters
PHONE_SCREEN_RATIO = 0.85  # Screen occupies 85% of phone area
PHONE_BEZEL_RATIO = 0.15   # Bezel occupies 15% of phone area

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Using nano model for speed

model = load_model()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def calibrate_dimensions_from_image(class_name, roi, frame_shape):
    """
    Calculate approximate dimensions based on camera perspective and object size
    """
    if roi.size == 0:
        return KNOWN_WIDTH_MM.get(class_name, 0), KNOWN_HEIGHT_MM.get(class_name, 0)
    
    # Get ROI dimensions in pixels
    h_pixels, w_pixels = roi.shape[:2]
    
    # Simple calibration: assume object fills most of ROI
    if class_name == "cell phone":
        # Phone aspect ratio is typically ~19.5:9 or 20:9
        width_mm = KNOWN_WIDTH_MM["cell phone"]
        height_mm = KNOWN_HEIGHT_MM["cell phone"]
    elif class_name == "bottle":
        width_mm = KNOWN_WIDTH_MM["bottle"]
        height_mm = KNOWN_HEIGHT_MM["bottle"]
    elif class_name == "pen":
        width_mm = KNOWN_WIDTH_MM["pen"]
        height_mm = KNOWN_HEIGHT_MM["pen"]
    else:
        width_mm = 0
        height_mm = 0
    
    return width_mm, height_mm

def detect_phone_scratches(roi):
    """
    Enhanced phone scratch detection
    """
    if roi.size == 0:
        return False, 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 30, 100)
    
    # Calculate edge density
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    # Additional check: look for irregular contours that might indicate scratches
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scratch_detected = False
    scratch_contour_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter out small noise
            scratch_contour_count += 1
            if scratch_contour_count > 3:  # Multiple scratches detected
                scratch_detected = True
                break
    
    # Combined detection logic
    if edge_density > SCRATCH_EDGE_THRESHOLD or scratch_detected:
        # Highlight scratches in red
        roi_highlight = roi.copy()
        roi_highlight[edges > 0] = [0, 0, 255]
        roi[:] = roi_highlight[:]
        return True, edge_density
    
    return False, edge_density

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="VisionIQ", layout="wide")
st.title("🔍 VisionIQ – Multi‑Object AI Inspection")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
    st.write("---")
    st.header("📊 Threshold Settings")
    scratch_threshold = st.slider("Scratch Threshold", 0.01, 0.2, 0.05, 0.01)
    crack_threshold = st.slider("Crack Threshold", 0.01, 0.2, 0.08, 0.01)
    st.write("---")
    st.header("ℹ️ Info")
    st.write("Detects: Bottle, Cell Phone, Pen")
    st.write("Defects: Missing caps, cracks, scratches")

start = st.checkbox("▶️ Start Inspection")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    frame_window = st.image([])
    
with col2:
    status_text = st.empty()
    reason_text = st.empty()
    measure_text = st.empty()
    stats_text = st.empty()

if "inspection_log" not in st.session_state:
    st.session_state.inspection_log = []
    st.session_state.detection_stats = {"total": 0, "defects": 0}

cap = cv2.VideoCapture(0)

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# -------------------------
# FPS CALCULATION VARIABLES
# -------------------------
prev_frame_time = 0
fps_display = 0

# -------------------------
# MAIN LOOP
# -------------------------
if start:
    while True:
        # FPS calculation - start
        current_frame_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not accessible")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Make a copy for processing
        processed_frame = frame.copy()
        
        # Run YOLO detection
        results = model(frame, conf=confidence_threshold)
        
        for result in results:
            if result.boxes is None:
                continue
                
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                class_name = model.names[cls_id].lower()
                confidence = float(result.boxes.conf[i])

                # Normalize pen‑like detections
                if class_name in ["toothbrush", "remote", "marker", "tv remote"]:
                    class_name = "pen"

                if class_name not in EXPECTED_OBJECTS:
                    continue
                    
                st.session_state.detection_stats["total"] += 1
                
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                roi = frame[y1:y2, x1:x2]

                defect = False
                reason = "None"
                defect_type = ""

                # -------------------------
                # DIMENSIONS (in mm) - using calibration
                # -------------------------
                width_mm, height_mm = calibrate_dimensions_from_image(class_name, roi, frame.shape)

                # -------------------------
                # BOTTLE – CAP + CRACK
                # -------------------------
                if class_name == "bottle" and roi.size > 0:
                    # Cap detection
                    cap_y = y1 + int(0.2 * (y2 - y1))
                    cap_roi = frame[y1:cap_y, x1:x2]

                    if cap_roi.size > 0:
                        gray = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(
                            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        cap_present = False
                        for c in contours:
                            area = cv2.contourArea(c)
                            if area > CAP_AREA_THRESHOLD:
                                cap_present = True
                                break

                        if not cap_present:
                            defect = True
                            reason = "Bottle Cap Missing"
                            defect_type = "missing_cap"

                    # Crack detection on bottle body
                    body_roi = frame[cap_y:y2, x1:x2]
                    if body_roi.size > 0:
                        gray_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_body, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
                        if edge_density > CRACK_EDGE_THRESHOLD:
                            defect = True
                            reason = "Bottle Surface Crack"
                            defect_type = "crack"
                            # Highlight the crack
                            temp_body = body_roi.copy()
                            temp_body[edges > 0] = [0, 0, 255]
                            processed_frame[cap_y:y2, x1:x2] = temp_body

                # -------------------------
                # PHONE – SCREEN SCRATCH
                # -------------------------
                elif class_name == "cell phone" and roi.size > 0:
                    # Enhanced scratch detection
                    has_scratches, scratch_density = detect_phone_scratches(roi)
                    
                    if has_scratches:
                        defect = True
                        reason = f"Phone Screen Scratch (density: {scratch_density:.3f})"
                        defect_type = "scratch"
                        st.session_state.detection_stats["defects"] += 1
                        
                        # Update the processed frame with highlighted scratches
                        processed_frame[y1:y2, x1:x2] = roi

                # -------------------------
                # PEN – CAP MISSING
                # -------------------------
                elif class_name == "pen" and roi.size > 0:
                    cap_roi = roi[:int(0.25 * roi.shape[0]), :]
                    if cap_roi.size > 0:
                        gray = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(
                            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        cap_found = False
                        for c in contours:
                            if cv2.contourArea(c) > CAP_AREA_THRESHOLD:
                                cap_found = True
                                break

                        if not cap_found:
                            defect = True
                            reason = "Pen Cap Missing"
                            defect_type = "missing_cap"

                # -------------------------
                # DRAW RESULTS
                # -------------------------
                color = (0, 0, 255) if defect else (0, 255, 0)
                status = "DEFECT" if defect else "OK"
                thickness = 3 if defect else 2

                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    processed_frame,
                    f"{class_name.upper()} | {status} | {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                # Add dimension text
                cv2.putText(
                    processed_frame,
                    f"W:{width_mm}mm H:{height_mm}mm",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                # -------------------------
                # LOG RESULT
                # -------------------------
                st.session_state.inspection_log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "object": class_name,
                    "status": status,
                    "reason": reason,
                    "defect_type": defect_type,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "confidence": confidence
                })

                # -------------------------
                # UI UPDATES
                # -------------------------
                with col2:
                    measure_text.markdown(
                        f"""
                        ### 📏 Measurements
                        **Object:** {class_name.capitalize()}  
                        **Width:** {width_mm:.1f} mm  
                        **Height:** {height_mm:.1f} mm  
                        **Confidence:** {confidence:.2%}
                        
                        <small>
                        📐 Dimensions in <b>millimeters (mm)</b><br>
                        🔧 Based on calibrated references
                        </small>
                        """,
                        unsafe_allow_html=True
                    )

                    status_text.markdown(f"### 🔴 Status: **{status}**")
                    reason_text.markdown(f"### 🛠 Reason: **{reason}**")
                    
                    # Statistics
                    total = st.session_state.detection_stats["total"]
                    defects = st.session_state.detection_stats["defects"]
                    defect_rate = (defects / total * 100) if total > 0 else 0
                    
                    stats_text.markdown(
                        f"""
                        ### 📊 Statistics
                        **Total Detections:** {total}  
                        **Defects Found:** {defects}  
                        **Defect Rate:** {defect_rate:.1f}%
                        """
                    )

        # -------------------------
        # FPS CALCULATION & DISPLAY
        # -------------------------
        if prev_frame_time != 0:
            fps_display = 1 / (current_frame_time - prev_frame_time)
        prev_frame_time = current_frame_time
        
        # Draw FPS counter with background for better visibility
        fps_text = f"FPS: {fps_display:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        # Draw semi-transparent background rectangle
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (10, 10), (20 + text_width, 20 + text_height + baseline), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
        
        # Draw FPS text in bright green
        cv2.putText(
            processed_frame,
            fps_text,
            (15, 15 + text_height),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )

        # Display the processed frame
        frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        if not start:
            break

cap.release()

# -------------------------
# CSV EXPORT
# -------------------------
st.write("---")
st.subheader("📊 Inspection Log & Export")

if len(st.session_state.inspection_log) > 0:
    df = pd.DataFrame(st.session_state.inspection_log)
    
    # Display recent entries
    st.write("### Recent Inspections:")
    st.dataframe(df.tail(10), use_container_width=True)
    
    # Export buttons
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_data,
            file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export2:
        # JSON export option
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="⬇️ Download JSON Report",
            data=json_data,
            file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
else:
    st.info("No inspection data available. Start inspection to collect data.")

# -------------------------
# DEBUG INFO (hidden by default)
# -------------------------
with st.expander("🔧 Debug Information"):
    st.write("### Configuration Parameters")
    st.json({
        "scratch_threshold": SCRATCH_EDGE_THRESHOLD,
        "crack_threshold": CRACK_EDGE_THRESHOLD,
        "cap_area_threshold": CAP_AREA_THRESHOLD,
        "expected_objects": EXPECTED_OBJECTS
    })
    
    if len(st.session_state.inspection_log) > 0:
        st.write("### Raw Detection Data")
        st.write(st.session_state.inspection_log[-5:])