🛰️ VisionIQ — AI‑Powered Object Inspection System
VisionIQ is a real‑time AI inspection platform that detects objects, identifies physical defects, and estimates dimensions using Computer Vision, YOLOv8, and ArUco marker‑based calibration.
The system was built and presented at AeroThon 5.0 (HAL – Hindustan Aeronautics Limited) where it received special appreciation from judges for practicality and innovation.

🚀 Key Features
Multi‑object detection using YOLOv8

Physical defect recognition (cracks, missing caps, broken parts, surface scratches, dents, etc.)

ArUco marker calibration for real‑time dimension measurements (mm)

Streamlit dashboard with analytics & status reporting

Works on webcam input for real‑time demos

Modular design for future drone, factory & warehouse deployment

🧠 Tech Stack
Layer	Tools / Models
Object Detection	YOLOv8
Calibration	OpenCV ArUco
Vision Processing	OpenCV, NumPy
UI / Dashboard	Streamlit
Backend Logic	Python
Analytics	Pandas
Hardware Input	USB/Webcam (extendable to LiDAR or Drone footage)
⚙️ How It Works
Capture live video stream

Detect objects using YOLOv8

Perform defect analysis per object type

Use ArUco markers to calculate physical dimensions

Display results + defect reasons + measurements in UI

Log session data for export and insights

🧩 Supported Objects & Defect Types
Object	Defect Examples
Bottle	Missing cap, cracks
Phone	Screen scratches
Pen	Missing cap
Suitcase	Broken handle, dent, wheel damage
Cup	Rim chip, handle damage
Chair	Broken leg, seat & frame damage
Clock	Missing hands, cracked glass
Laptop	Screen crack, keyboard damage
Backpack	Zipper tear, strap damage
Book	Spine damage, torn pages
📊 Live Outputs
✔ Dimensions (mm)
✔ Defect status (OK / DEFECT)
✔ Confidence levels
✔ Calibration ratio
✔ Exportable logs (CSV/JSON)

🧭 Future Scope
Drone-based industrial inspection

Warehouse automation for damaged goods

Integration with LiDAR depth maps (planned)

Autonomous robotics for manufacturing QA

Multi-camera calibration & 3D estimation

Edge deployment (Jetson Nano, Raspberry Pi 5)

🏅 Hackathon Context
Developed for AeroThon 5.0 — Hindustan Aeronautics Limited (HAL)
Received Special Appreciation from the panel for:

Innovation in inspection automation

Practical industrial use case

Real‑time working prototype

🧑‍💻 Team VisionIQ
Shlok Gawade • Shoumik Trivedi • Sahil Kanchan

📸 Demo
(Add screenshots / GIFs here after upload)

🏁 Getting Started
git clone <repo-link>
pip install -r requirements.txt
streamlit run app.py
Requires:

Python 3.9+

Webcam

(Optional) Printed ArUco marker for calibration
