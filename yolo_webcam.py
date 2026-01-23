detected_debug = []

for r in results:
    if r.boxes is not None:
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            detected_debug.append(model.names[cls_id])

cv2.putText(
    frame,
    f"Detected: {detected_debug}",
    (20, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)