# Gunakan image Python dasar yang spesifik dan stabil
FROM python:3.9-slim-buster

# Instal dependensi sistem yang dibutuhkan untuk dlib, opencv, dan psycopg2
# Ini harus dilakukan SEBELUM instalasi paket Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Buat direktori aplikasi di dalam container
WORKDIR /app

# Setel PYTHONPATH agar Python dapat menemukan modul Anda
ENV PYTHONPATH=/app:$PYTHONPATH

# Salin requirements.txt dan instal dependensi Python
# Ini adalah praktik terbaik untuk caching Docker: instal dependensi sebelum menyalin seluruh kode
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi Anda ke direktori kerja container
# Pastikan file main.py ada di your_project_root/api/main.py
COPY . .

# Perintah untuk menjalankan aplikasi menggunakan Gunicorn
# Formatnya adalah <module_path>:<callable_app_object>
# 'api.main' merujuk pada file 'main.py' di dalam direktori 'api',
# dan 'app' adalah objek aplikasi Flask yang didefinisikan di main.py.
CMD ["gunicorn", "api.main:app"]
