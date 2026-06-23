import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import sqlite3

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Parking Monitor", layout="wide")
st.title("🅿️ AI Car Park Monitor & Vehicle Locator")
st.markdown("Real-time vehicle detection, slot assignment, and license plate registry.")

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    """Creates the local database file and tables if they don't exist."""
    with sqlite3.connect("parking.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_registry (
                plate_number TEXT PRIMARY KEY,
                slot_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Run database setup immediately on startup
init_db()

# --- 3. VIRTUAL PARKING SLOTS CONFIGURATION ---
# These coordinates (x1, y1, x2, y2) represent pixels on your camera frame.
# Adjust these values based on your physical camera angle and frame size.
PARKING_SLOTS = {
    "Slot A1": (50, 80, 280, 420),
    "Slot A2": (320, 80, 550, 420),
    "Slot B1": (590, 80, 820, 420)
}

# COCO Dataset IDs for vehicles (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- 4. MODEL CACHING ---
@st.cache_resource
def load_models():
    """Loads deep learning models into memory once and retains them."""
    yolo_model = YOLO('yolov8n.pt')
    ocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA installed
    return yolo_model, ocr_reader

model, reader = load_models()

# --- 5. SIDEBAR: CUSTOMER KIOSK (FIND MY CAR) ---
st.sidebar.header("🔍 Find My Car")
search_query = st.sidebar.text_input("Enter your License Plate:").strip().upper()

if st.sidebar.button("Locate Vehicle"):
    if search_query:
        with sqlite3.connect("parking.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT slot_name, timestamp FROM vehicle_registry WHERE plate_number = ?", 
                (search_query,)
            )
            result = cursor.fetchone()
            
            if result:
                slot, time_parked = result
                st.sidebar.success(f"🚗 Vehicle found! Your car is in **{slot}**.")
                st.sidebar.caption(f"Registered since: {time_parked}")
            else:
                st.sidebar.error("Vehicle not found. Verify the plate characters and try again.")
    else:
        st.sidebar.warning("Please enter a valid license plate number.")

st.sidebar.markdown("---")

# --- 6. SIDEBAR: ADMIN MONITOR CONTROLS ---
st.sidebar.header("⚙️ Live Stream Configuration")
run_camera = st.sidebar.checkbox("🟢 Start Parking Camera")
confidence = st.sidebar.slider("YOLO Detection Confidence", 0.10, 1.00, 0.40, 0.05)
ocr_confidence = st.sidebar.slider("OCR Text Confidence", 0.10, 1.00, 0.30, 0.05)

# --- 7. MAIN VIDEO STREAMS AND LOGIC ---
frame_placeholder = st.empty()

if run_camera:
    cap = cv2.VideoCapture(0)  # Use 0 for built-in camera, or replace with an RTSP stream link
    
    if not cap.isOpened():
        st.error("Error: Unable to connect to camera source.")
    else:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to fetch frames from the camera.")
                break
            
            # Draw static virtual parking slots boundaries on screen
            for slot_name, (sx1, sy1, sx2, sy2) in PARKING_SLOTS.items():
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 255, 0), 2)
                cv2.putText(frame, slot_name, (sx1 + 5, sy1 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Process frame using YOLOv8
            results = model(frame, conf=confidence, classes=VEHICLE_CLASSES, stream=True)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Compute vehicle bounding box center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Render visual marker at vehicle's spatial center
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Determine if center point falls within predefined slot coordinates
                    current_slot = "Unknown"
                    for slot_name, (sx1, sy1, sx2, sy2) in PARKING_SLOTS.items():
                        if sx1 < center_x < sx2 and sy1 < center_y < sy2:
                            current_slot = slot_name
                            break

                    # Isolate crop bounds ensuring frame dimensional compliance
                    y1_crop, y2_crop = max(0, y1), min(frame.shape[0], y2)
                    x1_crop, x2_crop = max(0, x1), min(frame.shape[1], x2)
                    vehicle_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    if vehicle_crop.size > 0:
                        # Extract characters within the isolated target vehicle crop
                        ocr_result = reader.readtext(vehicle_crop)
                        
                        for (bbox, text, prob) in ocr_result:
                            # Clean string output stripping spaces and special symbols
                            cleaned_text = "".join(c for c in text if c.isalnum()).upper()
                            
                            # Filter strings ensuring they meet minimal length and confidence requirements
                            if prob > ocr_confidence and len(cleaned_text) >= 4:
                                
                                # Write data or overwrite existing entries within the SQLite file
                                if current_slot != "Unknown":
                                    with sqlite3.connect("parking.db") as conn:
                                        cursor = conn.cursor()
                                        cursor.execute('''
                                            REPLACE INTO vehicle_registry (plate_number, slot_name, timestamp)
                                            VALUES (?, ?, CURRENT_TIMESTAMP)
                                        ''', (cleaned_text, current_slot))
                                        conn.commit()
                                
                                # Apply tracking graphics on the output image
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{cleaned_text} -> {current_slot}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                break
                    else:
                        # Draw generic bounding box if text reading is unverified
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Reformat image array configurations from BGR to standard RGB colorspaces
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
        cap.release()
else:
    st.info("Ensure the administrative 'Start Parking Camera' checkbox is enabled to initialize processing loops.")
