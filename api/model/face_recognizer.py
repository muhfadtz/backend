import face_recognition
import os

# Path folder known faces, relatif terhadap file ini

KNOWN_FACE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'known_faces'
)


# Global variables yang akan menyimpan data wajah
known_encodings = []
known_names = []

def load_known_faces():
    """Muat semua wajah dan nama dari folder known_faces."""
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACE_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(KNOWN_FACE_DIR, filename)
            image = face_recognition.load_image_file(path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                name = os.path.splitext(filename)[0]
                try:
                    nim, nama = name.split('_', 1)
                    names.append({'nim': nim, 'nama': nama})
                except ValueError:
                    # Kalau format filename gak sesuai, skip
                    print(f"Warning: filename {filename} tidak sesuai format 'nim_nama.ext'")
    return encodings, names

# Load known faces saat modul ini diimport
known_encodings, known_names = load_known_faces()

def recognize_face(image):
    """Mengenali wajah dari gambar input dan mengembalikan data jika ditemukan."""
    unknown_encodings = face_recognition.face_encodings(image)
    if not unknown_encodings:
        return None

    unknown_encoding = unknown_encodings[0]
    results = face_recognition.compare_faces(known_encodings, unknown_encoding)
    if True in results:
        match_index = results.index(True)
        return known_names[match_index]
    return None

def reload_faces():
    """Fungsi untuk reload known faces saat ada wajah baru yang didaftarkan."""
    global known_encodings, known_names
    known_encodings, known_names = load_known_faces()

