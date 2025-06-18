from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import base64
import cv2
import numpy as np
import os
import re
import face_recognition
import bcrypt

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# List of allowed origins (frontend URLs)
origins = [
    "https://face-recognition-attendance-mx7j-9t2fva34u.vercel.app"
]

# Allow CORS for credentials and custom origins
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

# Tambahkan header CORS ke semua response
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

# Konfigurasi database
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
            sslmode='require'
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

@app.route('/attendance', methods=['POST', 'OPTIONS'])
def attendance():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image_b64 = data['image'].split(',')[-1]
        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_cv2 is None:
            return jsonify({'error': 'Invalid image data'}), 400

        current_face_locations = face_recognition.face_locations(image_cv2)
        if not current_face_locations:
            return jsonify({'error': 'No face detected in image'}), 404

        current_face_encodings = face_recognition.face_encodings(image_cv2, current_face_locations)
        if not current_face_encodings:
            return jsonify({'error': 'Encoding failed'}), 400

        current_face_encoding = current_face_encodings[0]
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 503

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT nip, nama, face_embedding FROM \"Karyawan\" WHERE face_embedding IS NOT NULL")
        rows = cursor.fetchall()

        known_encodings = []
        known_nips = []
        known_names = []

        for row in rows:
            try:
                encoding = np.array([float(x) for x in row['face_embedding'].strip('[]').split(',')])
                known_encodings.append(encoding)
                known_nips.append(row['nip'])
                known_names.append(row['nama'])
            except:
                continue

        matches = face_recognition.compare_faces(known_encodings, current_face_encoding, tolerance=0.5)
        distances = face_recognition.face_distance(known_encodings, current_face_encoding)

        if not distances.any():
            return jsonify({'error': 'No known faces to compare'}), 500

        best_match_index = np.argmin(distances)
        if matches[best_match_index]:
            return jsonify({
                'message': 'Face recognized',
                'name': known_names[best_match_index],
                'nip': known_nips[best_match_index]
            }), 200
        else:
            return jsonify({'error': 'Face not recognized'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/register-face', methods=['POST', 'OPTIONS'])
def register_face():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    if not data or 'nip' not in data or 'fotoWajah' not in data:
        return jsonify({'error': 'Missing NIP or fotoWajah'}), 400
    try:
        image_b64 = data['fotoWajah'].split(',')[-1]
        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        face_locations = face_recognition.face_locations(image_cv2)
        if not face_locations:
            return jsonify({'error': 'No face detected'}), 400
        if len(face_locations) > 1:
            return jsonify({'error': 'Multiple faces detected'}), 400

        encoding = face_recognition.face_encodings(image_cv2, face_locations)[0]
        encoding_list = [float(x) for x in encoding]
        return jsonify({'face_encoding': encoding_list, 'message': 'Face encoded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')

    if email == admin_email and password == admin_password:
        return jsonify({"message": "Login berhasil", "role": "admin"}), 200

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection error"}), 503
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT nama, password FROM public.\"Karyawan\" WHERE email = %s", (email,))
        row = cursor.fetchone()
        if row:
            nama, hashed_password = row
            if bcrypt.checkpw(password.encode(), hashed_password.encode()):
                return jsonify({"message": "Login berhasil", "role": "user", "nama": nama}), 200
        return jsonify({"message": "Email atau password salah"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
