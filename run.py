from flask import Flask, render_template, Response, jsonify, redirect, url_for, request, flash
import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import numpy as np
import time
import threading
import atexit
import uuid
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Enhanced Flask-SocketIO Setup with explicit protocol version
socketio = SocketIO(app,
                  cors_allowed_origins="*",
                  async_mode='threading',
                  engineio_logger=True,
                  logger=True,
                  ping_timeout=60,
                  ping_interval=25,
                  max_http_buffer_size=10 * 1024 * 1024,
                  allow_upgrades=True,
                  http_compression=True,
                  engineio_version=4)  # Explicit Engine.IO v4

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# YOLO & Camera Setup
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
lock = threading.Lock()
latest_detections = []
latest_barcodes = []
latest_fps = 0.0
camera_active = True
latest_frame = None
frame_lock = threading.Lock()

# User tracking
active_users = set()
active_users_lock = threading.Lock()

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(8), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    national_id = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Chat Message Model
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(8), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def track_active_users():
    if current_user.is_authenticated:
        with active_users_lock:
            active_users.add(current_user.user_id)

@app.teardown_request
def remove_active_user(exception=None):
    if current_user.is_authenticated:
        with active_users_lock:
            active_users.discard(current_user.user_id)

def release_camera():
    global cap
    if cap.isOpened():
        cap.release()
        logger.info("Camera released")

atexit.register(release_camera)

def process_frame(frame):
    global model, lock, latest_detections, latest_barcodes
    
    # Object detection
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Code detection (both QR and barcodes)
    codes = decode(frame)
    codes_data = []
    
    for code in codes:
        code_data = code.data.decode('utf-8')
        code_type = code.type
        codes_data.append({"data": code_data, "type": code_type})
        
        # Different colors for QR codes vs barcodes
        color = (0, 255, 0)  # Green for barcodes
        if code_type == "QRCODE":
            color = (255, 0, 0)  # Blue for QR codes
            
        rect_points = code.polygon
        if rect_points:
            pts = np.array([rect_points], np.int32)
            cv2.polylines(annotated_frame, [pts], True, color, 2)
            
        x, y, w, h = code.rect
        cv2.putText(annotated_frame, f"{code_type}: {code_data}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    
    # Process detection data
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
    
    with lock:
        latest_detections = detection_data
        latest_barcodes = codes_data
    
    return annotated_frame

def generate_frames():
    global latest_frame, camera_active, cap, latest_fps, frame_lock
    prev_time = time.time()
    
    while True:
        if not camera_active:
            time.sleep(0.1)
            continue
            
        # Priority to SocketIO frames if available
        with frame_lock:
            if latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
                time.sleep(0.05)
                continue
                
        # Fallback to webcam if no SocketIO frames
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        # Process frame and calculate FPS
        annotated_frame = process_frame(frame)
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        latest_fps = round(fps, 2)
        
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# SocketIO Events with protocol version checking
@socketio.on('connect')
def handle_connect():
    try:
        transport = request.args.get('transport')
        eio = request.args.get('EIO')
        
        logger.info(f"Connection attempt - SID: {request.sid}, EIO: {eio}, Transport: {transport}")
        
        if eio != '4':
            logger.warning(f"Rejected connection - Unsupported Engine.IO version: {eio}")
            return False
            
        if current_user.is_authenticated:
            emit('auth_status', {'authenticated': True, 'user': current_user.first_name})
            with active_users_lock:
                active_users.add(current_user.user_id)
            logger.info(f"Authenticated client connected: {current_user.user_id}")
        else:
            emit('auth_status', {'authenticated': False})
            logger.info("Unauthenticated client connected")
            
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        return False

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    if current_user.is_authenticated:
        with active_users_lock:
            active_users.discard(current_user.user_id)

# Protocol information endpoint
@app.route('/socket.io/')
def socketio_info():
    return jsonify({
        "status": "ready",
        "socketio": True,
        "engineio_version": "4",
        "supported_transports": ["websocket", "polling"],
        "ping_interval": 25000,
        "ping_timeout": 60000
    }), 200

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "connected" if db.engine else "disconnected",
            "camera": "active" if cap.isOpened() else "inactive",
            "model": "loaded"
        }
    })

# Handle frames from Android devices
@socketio.on('android_frame')
def handle_android_frame(data):
    global latest_frame
    try:
        logger.info(f"Received frame from Android (size: {len(data) if isinstance(data, str) else 'binary'})")
        
        # Handle both JSON and direct base64 strings
        img_data = base64.b64decode(data['data'].split(",")[1] if isinstance(data, dict) else data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image frame")

        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        
        with frame_lock:
            latest_frame = buffer.tobytes()
        
        emit('processed_frame', {
            'status': 'success',
            'detections': latest_detections,
            'barcodes': latest_barcodes,
            'fps': latest_fps,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info("Processed Android frame successfully")
        
    except Exception as e:
        logger.error(f"Android frame handling error: {str(e)}")
        emit('processed_frame', {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })

# Handle frames from web clients
@socketio.on('frame')
def handle_frame(data):
    handle_android_frame(data)

# --- Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        national_id = request.form.get('national_id')

        # Validate required fields
        if not all([email, password, confirm_password, first_name, last_name, national_id]):
            flash('All fields are required')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))

        new_user = User(
            user_id=str(uuid.uuid4())[:8],
            first_name=first_name,
            last_name=last_name,
            national_id=national_id,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')
    
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('reset_password'))

        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash('Your password has been updated. You can now log in.')
            return redirect(url_for('login'))
        else:
            flash('No account found with that email')
    
    return render_template('reset_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/video')
@login_required
def video():
    return Response(generate_frames(), 
                  mimetype='multipart/x-mixed-replace; boundary=frame',
                  headers={
                      'Cache-Control': 'no-cache, no-store, must-revalidate',
                      'Pragma': 'no-cache',
                      'Expires': '0'
                  })

@app.route('/detections')
@login_required
def get_detections():
    with lock:
        return jsonify({
            "objects": latest_detections,
            "barcodes": latest_barcodes,
            "fps": latest_fps
        })

@app.route('/camera/status')
@login_required
def camera_status():
    return jsonify({"active": camera_active})

@app.route('/camera/toggle', methods=['POST'])
@login_required
def toggle_camera():
    global camera_active, cap
    camera_active = not camera_active
    
    if not camera_active and cap.isOpened():
        cap.release()
        logger.info("Camera released due to toggle")
    elif camera_active and not cap.isOpened():
        cap = cv2.VideoCapture(0)
        logger.info("Camera reinitialized due to toggle")
    
    return jsonify({"active": camera_active})

@app.route('/active_users')
@login_required
def get_active_users():
    with active_users_lock:
        users = User.query.filter(User.user_id.in_(list(active_users))).all()
        return jsonify([{
            'user_id': u.user_id,
            'name': f"{u.first_name} {u.last_name}"
        } for u in users])

@app.route('/chat/send', methods=['POST'])
@login_required
def send_chat():
    message = request.form.get('message')
    if message and len(message.strip()) > 0:
        new_message = ChatMessage(
            user_id=current_user.user_id,
            message=message.strip(),
            first_name=current_user.first_name,
            last_name=current_user.last_name
        )
        db.session.add(new_message)
        db.session.commit()
        socketio.emit('new_message', {
            'user_id': current_user.user_id,
            'name': f"{current_user.first_name} {current_user.last_name}",
            'message': message.strip(),
            'timestamp': datetime.utcnow().strftime("%H:%M:%S")
        })
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Empty message"}), 400

@app.route('/chat/messages')
@login_required
def get_chat_messages():
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    messages = ChatMessage.query.filter(
        ChatMessage.timestamp >= one_hour_ago
    ).order_by(ChatMessage.timestamp.desc()).limit(50).all()
    
    messages_data = [{
        "id": msg.id,
        "user_id": msg.user_id,
        "name": f"{msg.first_name} {msg.last_name}",
        "message": msg.message,
        "timestamp": msg.timestamp.strftime("%H:%M:%S")
    } for msg in messages]
    return jsonify(messages_data[::-1])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    logger.info("Starting server with Engine.IO v4 support...")
    socketio.run(app, 
                 host='0.0.0.0', 
                 port=5000, 
                 debug=True,
                 allow_unsafe_werkzeug=True,
                 use_reloader=False)