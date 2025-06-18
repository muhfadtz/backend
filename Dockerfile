# Gunakan image Python dasar. Pilih versi Python 3.9 (paling stabil untuk dlib)
FROM python:3.9-slim-buster 

# Instal dependensi sistem yang dibutuhkan dlib, opencv, dan psycopg2 (jika akan dikembalikan)
# Ini harus dilakukan SEBELUM instalasi pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    # Tambahkan libpq-dev HANYA JIKA Anda akan mengembalikan psycopg2-binary
    # libpq-dev \ 
    && rm -rf /var/lib/apt/lists/*

# Buat direktori aplikasi di dalam container
WORKDIR /app

# Salin requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi Anda
COPY . .

# Perintah untuk menjalankan aplikasi (gunakan gunicorn)
CMD ["gunicorn", "api.main:app"]
