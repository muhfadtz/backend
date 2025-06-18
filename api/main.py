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

# --- [KONFIGURASI CORS] ---
# Daftar URL frontend yang diizinkan untuk terhubung
origins = [
    "https://facerecognition-attendance-production.up.railway.app",
    "https://face-recognition-attendance-mx7j-9t2fva34u.vercel.app",
    "http://localhost:3000" # Untuk development lokal
]
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

# --- [DATABASE CONNECTION] ---
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

# --- [FUNGSI PEMROSESAN GAMBAR DENGAN LOGGING LENGKAP UNTUK DIAGNOSIS] ---
def process_image_for_face_recognition(base64_string):
    """
    Fungsi terpusat untuk memproses gambar dari string base64.
    Menerapkan perbaikan untuk padding, konversi warna (RGBA -> BGR), dan resize.
    Mengembalikan objek gambar yang siap untuk face recognition atau (None, error_message).
    """
    app.logger.info("--- BUKTI: Memasuki fungsi process_image_for_face_recognition ---")

    # 1. Perbaiki padding base64
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    
    # 2. Dekode string base64
    try:
        image_data = base64.b64decode(base64_string)
    except base64.binascii.Error as e:
        app.logger.error(f"Base64 decoding failed: {e}")
        return None, "Invalid base64 string provided."

    np_arr = np.frombuffer(image_data, np.uint8)
    
    # 3. Muat gambar
    image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if image_cv2 is None:
        app.logger.error("--- BUKTI: cv2.imdecode MENGEMBALIKAN None! ---")
        return None, "Invalid image data, could not be decoded."

    # --- BUKTI PENTING ---
    # Log bentuk (shape) gambar SEBELUM konversi
    try:
        app.logger.info(f"--- BUKTI: Shape gambar SETELAH decode: {image_cv2.shape} ---")
    except AttributeError:
        app.logger.error("--- BUKTI: Gagal mendapatkan shape gambar, objek gambar tidak valid. ---")
        return None, "Invalid image object after decoding."

    # 4. Konversi gambar 4-channel (RGBA) ke 3-channel (BGR)
    try:
        # Pengecekan shape[2] hanya valid jika gambar punya lebih dari 2 dimensi (bukan grayscale murni)
        if len(image_cv2.shape) > 2 and image_cv2.shape[2] == 4:
            app.logger.info("--- BUKTI: Mendeteksi 4 channel, MENCOBA konversi BGRA ke BGR... ---")
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGRA2BGR)
            app.logger.info(f"--- BUKTI: Konversi BGRA ke BGR BERHASIL. Shape baru: {image_cv2.shape} ---")
    except IndexError:
        app.logger.info("--- BUKTI: Mendeteksi gambar Grayscale atau 3-channel, tidak ada konversi warna. ---")
        pass
    except Exception as e:
        app.logger.error(f"--- BUKTI: Terjadi error SAAT KONVERSI WARNA: {e} ---", exc_info=True)
        # Jangan hentikan proses, biarkan face_recognition yang gagal agar errornya sama
        pass

    # 5. Resize gambar
    MAX_WIDTH = 800
    height, width, *_ = image_cv2.shape
    if width > MAX_WIDTH:
        app.logger.info("--- BUKTI: Melakukan resize gambar... ---")
        ratio = MAX_WIDTH / float(width)
        new_height = int(height * ratio)
        image_cv2 = cv2.resize(image_cv2, (MAX_WIDTH, new_height), interpolation=cv2.INTER_AREA)
        app.logger.info(f"--- BUKTI: Resize BERHASIL. Shape akhir: {image_cv2.shape} ---")
        
    app.logger.info("--- BUKTI: Fungsi process_image_for_face_recognition SELESAI. Mengembalikan gambar. ---")
    return image_cv2, None

# --- [ENDPOINTS API] ---

@app.route('/attendance', methods=['POST', 'OPTIONS'])
def attendance():
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
        
    try:
        # Proses gambar menggunakan fungsi terpusat
        image_b64 = data['image'].split(',')[-1]
        image_cv2, error = process_image_for_face_recognition(image_b64)
        if error:
            return jsonify({'error': error}), 400

        # Lanjutkan dengan face recognition
        current_face_locations = face_recognition.face_locations(image_cv2)
        if not current_face_locations:
            return jsonify({'error': 'No face detected in the image'}), 404

        current_face_encodings = face_recognition.face_encodings(image_cv2, current_face_locations)
        if not current_face_encodings:
            return jsonify({'error': 'Failed to create face encoding'}), 400

        current_face_encoding = current_face_encodings[0]
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
                except (ValueError, AttributeError):
                    app.logger.warning(f"Skipping invalid embedding for NIP {row.get('nip', 'N/A')}")
                    continue
            
            if not known_encodings:
                 return jsonify({'error': 'No known faces found in database to compare with'}), 404

            matches = face_recognition.compare_faces(known_encodings, current_face_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, current_face_encoding)

            if len(distances) == 0:
                return jsonify({'error': 'Face distance calculation failed'}), 500

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
        app.logger.error(f"An unexpected error occurred in /attendance: {e}", exc_info=True)
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500
    finally:
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()

@app.route('/register-face', methods=['POST', 'OPTIONS'])
def register_face():
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json()
    if not data or 'nip' not in data or 'fotoWajah' not in data:
        return jsonify({'error': 'Missing NIP or fotoWajah'}), 400
        
    try:
        # Gunakan kembali fungsi pemrosesan gambar yang sudah robust
        image_b64 = data['fotoWajah'].split(',')[-1]
        image_cv2, error = process_image_for_face_recognition(image_b64)
        if error:
            return jsonify({'error': error}), 400

        face_locations = face_recognition.face_locations(image_cv2)
        if not face_locations:
            return jsonify({'error': 'No face detected'}), 400
        if len(face_locations) > 1:
            return jsonify({'error': 'Multiple faces detected, please use a photo with one face'}), 400

        encoding = face_recognition.face_encodings(image_cv2, face_locations)[0]
        encoding_list = [float(x) for x in encoding]
        return jsonify({'face_encoding': encoding_list, 'message': 'Face encoded successfully'}), 200
        
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /register-face: {e}", exc_info=True)
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

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
            if row and row.get('password'):
                hashed_password_from_db = row['password']
                # Bcrypt Bekerja dengan bytes, jadi kita perlu encode
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password_from_db.encode('utf-8')):
                    return jsonify({"message": "Login berhasil", "role": "user", "nama": row['nama']}), 200
            
            return jsonify({"message": "Email atau password salah"}), 401
            
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /api/login: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    finally:
        if conn and not conn.closed:
            conn.close()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    # Port diatur oleh environment variable PORT di Railway, atau 5000 jika dijalankan lokal
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False untuk produksi
