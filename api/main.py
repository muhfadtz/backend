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

# Inisialisasi Flask app
app = Flask(__name__)

# --- [PERBAIKAN KRITIS PADA KONFIGURASI CORS] ---
# Daftar alamat (origin) yang kita percayai untuk terhubung ke backend ini.
# Alamat ini HARUS alamat frontend Anda, bukan backend.
origins = [
    "https://face-recognition-attendance-mx7j.vercel.app/"
]
# Inisialisasi CORS dengan daftar origin yang benar dan spesifik
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)


# --- Menangani Preflight Request (OPTIONS) secara manual ---
@app.before_request
def handle_preflight():
    """
    Fungsi ini berjalan sebelum setiap request.
    Jika request adalah preflight (OPTIONS), kita akan merespons dengan header CORS
    yang benar dan menghentikan pemrosesan lebih lanjut.
    """
    if request.method.upper() == 'OPTIONS':
        # Dapatkan alamat origin dari request yang masuk
        origin = request.headers.get('Origin')
        
        # Hanya izinkan jika origin ada di dalam daftar yang kita percayai
        if origin in origins:
            response = app.make_default_options_response()
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, PUT, DELETE, PATCH'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            response.headers['Access-Control-Max-Age'] = 86400  # Cache preflight response
            return response
    # Jika bukan request OPTIONS, lanjutkan seperti biasa
    pass
# --- Akhir dari Perbaikan ---


# Konfigurasi Database
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def get_db_connection():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT]):
        app.logger.error("Flask: Database environment variables are not fully set.")
        return None
    try:
        app.logger.info(f"Flask: Attempting to connect to host '{DB_HOST}' on port '{DB_PORT}'...")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
            sslmode='require'
        )
        app.logger.info("Flask: Database connection successful.")
        return conn
    except psycopg2.Error as e:
        app.logger.error(f"Flask: Error connecting to PostgreSQL: {e}")
        return None

def sanitize_filename(name):
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '_', name)
    return name.lower()

@app.route('/attendance', methods=['POST'])
def attendance():
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
            return jsonify({'error': 'No face detected in current image'}), 404
        current_face_encodings = face_recognition.face_encodings(image_cv2, current_face_locations)
        if not current_face_encodings:
            return jsonify({'error': 'Could not create encoding for the current image'}), 400
        current_face_encoding = current_face_encodings[0]
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Flask: Could not connect to database for attendance.'}), 503
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        known_face_encodings_from_db = []
        known_nips_from_db = []
        known_names_from_db = []
        try:
            cursor.execute("SELECT nip, nama, face_embedding FROM \"Karyawan\" WHERE face_embedding IS NOT NULL")
            all_karyawan_with_embeddings = cursor.fetchall()
            for row in all_karyawan_with_embeddings:
                try:
                    str_encoding = row['face_embedding']
                    float_list = [float(x) for x in str_encoding.strip('[]').split(',')]
                    known_face_encodings_from_db.append(np.array(float_list))
                    known_nips_from_db.append(row['nip'])
                    known_names_from_db.append(row['nama'])
                except (ValueError, TypeError) as e:
                    app.logger.warning(f"Flask: Skipping NIP {row['nip']} due to parsing error in face_embedding: {e}")
                    continue
            if not known_face_encodings_from_db:
                   return jsonify({'error': 'No known faces with valid embeddings found in database'}), 404
            matches = face_recognition.compare_faces(known_face_encodings_from_db, current_face_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_face_encodings_from_db, current_face_encoding)
            if len(distances) == 0:
                return jsonify({'error': 'Face distance calculation failed or no known faces to compare against.'}), 500
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                return jsonify({
                    'message': 'Face recognized',
                    'name': known_names_from_db[best_match_index],
                    'nip': known_nips_from_db[best_match_index]
                }), 200
            else:
                return jsonify({'error': 'Face not recognized'}), 404
        except psycopg2.Error as db_err:
            app.logger.error(f"Flask: Database error during attendance check: {str(db_err)}")
            return jsonify({'error': f'Flask Database error: {str(db_err)}'}), 500
        finally:
            if conn:
                cursor.close()
                conn.close()
    except Exception as e:
        app.logger.error(f"Flask: Error in /attendance endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/register-face', methods=['POST'])
def register_face():
    data = request.get_json()
    if not data or 'nip' not in data or 'fotoWajah' not in data:
        return jsonify({'error': 'Data tidak lengkap: nip dan fotoWajah wajib diisi'}), 400
    try:
        image_b64_data_url = data['fotoWajah']
        if ',' not in image_b64_data_url:
            return jsonify({'error': 'Format data URL gambar tidak valid'}), 400
        header, image_b64 = image_b64_data_url.split(',', 1)
        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_cv2 is None:
            return jsonify({'error': 'Gagal decode gambar (Flask)'}), 400
        face_locations = face_recognition.face_locations(image_cv2)
        if not face_locations:
            return jsonify({'error': 'Tidak ada wajah terdeteksi (Flask)'}), 400
        if len(face_locations) > 1:
            return jsonify({'error': 'Terdeteksi lebih dari satu wajah. Harap gunakan foto satu wajah.'}), 400
        face_encoding_array = face_recognition.face_encodings(image_cv2, face_locations)[0]
        face_encoding_list = [float(val) for val in face_encoding_array]
        return jsonify({
            'face_encoding': face_encoding_list,
            'message': 'Wajah berhasil dikenali dan diencode'
        }), 200
    except Exception as e:
        app.logger.error(f"Flask error di /register-face: {str(e)}")
        return jsonify({'error': f'Flask error: {str(e)}'}), 500

@app.route('api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')
    if email == admin_email and password == admin_password:
        return jsonify({"message": "Login berhasil", "role": "admin"}), 200
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Tidak bisa konek ke database"}), 503
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT nama, password FROM public."Karyawan"
            WHERE email = %s
        """, (email,))
        row = cursor.fetchone()
        if row:
            nama, hashed_password = row
            if hashed_password and bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                return jsonify({
                    "message": "Login berhasil",
                    "role": "user",
                    "nama": nama
                }), 200
            else:
                return jsonify({"message": "Email atau password salah"}), 401
        else:
            return jsonify({"message": "Email atau password salah"}), 401
    except psycopg2.Error as db_err:
        return jsonify({"error": f"Database error: {str(db_err)}"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Flask server is running'}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available endpoints:")
    print("- POST /attendance - Face recognition attendance")
    print("- POST /register-face - Register new face")
    print("- POST /api/login - User login")
    print("- GET /health - Health check")
    app.run(host='0.0.0.0', port=5000, debug=True)
