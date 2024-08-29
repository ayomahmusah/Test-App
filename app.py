from flask import Flask, request, jsonify, render_template, Response, redirect, url_for
import cv2
import torch
import numpy as np
import base64
import io
import os
import time
from PIL import Image
from collections import Counter
from flask_mysqldb import MySQL
import MySQLdb.cursors
import MySQLdb.cursors, re, hashlib
import mysql.connector
from werkzeug.security import check_password_hash


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from resnet50 import ResNet50


app = Flask(__name__)
 
 
# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='12345',
    database='mydatabase'
)
cursor = conn.cursor()


checkpoint = torch.load("./checkpoints/best-checkpoint-009.pth", map_location=torch.device('cpu'))
ResNet50.load_state_dict(checkpoint["model_state_dict"])
ResNet50.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 768)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index1')
def sign_up():
    return render_template('index1.html')

@app.route('/home')
def home():
    # return render_template('home.html')
    return render_template('home.html')


@app.route('/prediction')
def predict():
    return render_template('prediction.html')

def hash_password(password):
    """ Hash a password using SHA-256 with a salt """
    salt = b'encrypt'  # Generate a new salt
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return pwd_hash.hex()

def verify_password(stored_password_hash, provided_password):
    """ Verify a stored password hash against a provided password """
    salt = b'encrypt' 
    stored_pwd_hash = stored_password_hash
    pwd_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return pwd_hash.hex() == stored_pwd_hash

@app.route('/register', methods=['GET', 'POST'])
def register():
      if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not name or not email or not password:
            return "Error: All fields are required!"
        
        # Hash the password
        hashed_password = hash_password(password)

        # Insert into the database
        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, hashed_password))
        conn.commit()
        
        return render_template('/home.html')
        
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return "Error: All fields are required!"

        # Fetch user record from database
        cursor.execute('SELECT password FROM users WHERE email = %s', (email,))
        result = cursor.fetchone()
        print(result)

        if result:
            hashed_password = result[0]

            # Check if the password matches
            if verify_password(hashed_password, password):
                # Redirect to the home page or a protected route
                print('verified')
                return redirect(url_for('home'))
            else:
                return "Error: Invalid email or password!"
        else:
            return "Error: No user found with this email!"

    return render_template('index.html')  # Adjust path if necessary


@app.route('/predict_from_video', methods=['POST'])
def predict_from_video():
    file = request.files['file']
    video_bytes = file.read()
    video_path = "temp_video.webm"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video file"}), 500

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reported total frames in video: {total_frames}")

    # Count frames manually and collect predictions
    frame_count = 0
    predictions = []
    num_samples = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Only process every nth frame to get approximately 5 samples
        if frame_count % max(1, (total_frames // num_samples)) == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            emotion = predict_emotion(pil_img, ResNet50)
            predictions.append(emotion)
            print(f"Processed frame {frame_count}, predicted emotion: {emotion}")
        
        if len(predictions) >= num_samples:
            break

    cap.release()
    os.remove(video_path)

    print(f"Actually processed frames: {frame_count}")
    print(f"Number of predictions made: {len(predictions)}")

    if not predictions:
        return jsonify({"error": "No frames were successfully processed"}), 500

    # Calculate the mode (most common prediction)
    pred_counts = Counter(predictions)
    final_emotion = pred_counts.most_common(1)[0][0]
    print("Final Predicted Emotion: ", final_emotion)
    return jsonify({"pred_emotion": final_emotion})


@app.route('/predict_from_image', methods=['POST'])
def predict_from_image():
    file = request.files['file']
    img_bytes = file.read()

    pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    emotion = predict_emotion(pil_image, ResNet50)
    print("Predicted Emotion: ", emotion)
    
    return jsonify({"pred_emotion": emotion})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def predict_emotion(image, model):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    
    emotions = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad','Surprised']
    return emotions[pred]


if __name__ == '__main__':
    app.run(debug=True)
