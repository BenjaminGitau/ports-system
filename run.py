from flask import Flask, render_template, Response, jsonify, url_for
import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import numpy as np
import time
import threading
import atexit

app = Flask(__name__)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
lock = threading.Lock()

latest_detections = []
latest_barcodes = []
latest_fps = 0.0

def release_camera():
    cap.release()

atexit.register(release_camera)

def generate_frames():
    global latest_detections, latest_barcodes, latest_fps
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Barcode detection
        barcodes = decode(frame)
        barcodes_data = []

        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            barcodes_data.append({
                "data": barcode_data,
                "type": barcode_type
            })

            rect_points = barcode.polygon
            if rect_points:
                pts = np.array([rect_points], np.int32)
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)

            x, y, w, h = barcode.rect
            cv2.putText(annotated_frame, f"{barcode_type}: {barcode_data}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Object detection
        labels = results[0].names
        detections = results[0].boxes
        detection_data = []
        for box in detections:
            conf = float(box.conf[0])
            if conf > 0.3:
                cls_id = int(box.cls[0])
                detection_data.append({
                    "label": labels[cls_id],
                    "confidence": round(conf * 100, 2)
                })

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Update globals safely
        with lock:
            latest_detections = detection_data
            latest_barcodes = barcodes_data
            latest_fps = round(fps, 2)

        # Stream frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    with lock:
        return jsonify({
            "objects": latest_detections,
            "barcodes": latest_barcodes,
            "fps": latest_fps
        })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
