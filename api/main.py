from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import base64
import cv2
import numpy as np
import os
import face_recognition
import bcrypt

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# --- Konfigurasi CORS ---
origins = [
    "https://facerecognition-attendance-production.up.railway.app",
    "https://face-recognition-attendance-mx7j-9t2fva34u.vercel.app",
    "http://localhost:3000"
]
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

# --- Konfigurasi koneksi database ---
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
        app.logger.error(f"Error connecting to PostgreSQL: {e}")
        return None

# --- Fungsi pemrosesan gambar ---
def process_image_for_face_recognition(base64_string):
    try:
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)

        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if image_cv2 is None:
            return None, "Image decoding failed"

        if len(image_cv2.shape) == 3 and image_cv2.shape[2] == 4:
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGRA2BGR)

        MAX_WIDTH = 800
        height, width = image_cv2.shape[:2]
        if width > MAX_WIDTH:
            ratio = MAX_WIDTH / float(width)
            new_height = int(height * ratio)
            image_cv2 = cv2.resize(image_cv2, (MAX_WIDTH, new_height), interpolation=cv2.INTER_AREA)

        return image_cv2, None

    except Exception as e:
        app.logger.error(f"Error in image processing: {e}", exc_info=True)
        return None, str(e)

# --- API Endpoint: Absensi ---
@app.route('/attendance', methods=['POST', 'OPTIONS'])
def attendance():
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_b64 = data['image'].split(',')[-1]
        image_cv2, error = process_image_for_face_recognition(image_b64)
        if error:
            return jsonify({'error': error}), 400

        face_locations = face_recognition.face_locations(image_cv2)
        if not face_locations:
            return jsonify({'error': 'No face detected'}), 404

        face_encodings = face_recognition.face_encodings(image_cv2, face_locations)
        if not face_encodings:
            return jsonify({'error': 'Face encoding failed'}), 400

        current_encoding = face_encodings[0]

        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 503

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
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

            if not known_encodings:
                return jsonify({'error': 'No known faces to compare'}), 404

            matches = face_recognition.compare_faces(known_encodings, current_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, current_encoding)

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
        app.logger.error(f"Attendance error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()

# --- API Endpoint: Register wajah ---
@app.route('/register-face', methods=['POST', 'OPTIONS'])
def register_face():
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json()
    if not data or 'nip' not in data or 'fotoWajah' not in data:
        return jsonify({'error': 'Missing data'}), 400

    try:
        image_b64 = data['fotoWajah'].split(',')[-1]
        image_cv2, error = process_image_for_face_recognition(image_b64)
        if error:
            return jsonify({'error': error}), 400

        face_locations = face_recognition.face_locations(image_cv2)
        if len(face_locations) != 1:
            return jsonify({'error': 'Harap gunakan foto dengan 1 wajah saja'}), 400

        encoding = face_recognition.face_encodings(image_cv2, face_locations)[0]
        encoding_list = [float(x) for x in encoding]

        return jsonify({'face_encoding': encoding_list, 'message': 'Face encoded successfully'}), 200

    except Exception as e:
        app.logger.error(f"Register-face error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

# --- API Endpoint: Login ---
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message": "Email dan password wajib diisi"}), 400

    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')

    if email == admin_email and password == admin_password:
        return jsonify({"message": "Login berhasil", "role": "admin"}), 200

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection error"}), 503

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT nama, password FROM public.\"Karyawan\" WHERE email = %s", (email,))
            row = cursor.fetchone()
            if row and bcrypt.checkpw(password.encode(), row['password'].encode()):
                return jsonify({"message": "Login berhasil", "role": "user", "nama": row['nama']}), 200
            return jsonify({"message": "Email atau password salah"}), 401
    except Exception as e:
        app.logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()

# --- API Endpoint: Health check ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200

# --- Main ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
