# LANGKAH 1: Pilih Sistem Operasi + Versi Python
# Kita gunakan python 3.9 yang stabil dan ringan
FROM python:3.9-slim

# LANGKAH 2: INSTALL SEMUA PERKAKAS SISTEM (INI BAGIAN KUNCINYA)
# Perintah ini akan dijalankan di dalam server Vercel
RUN apt-get update && apt-get install -y build-essential cmake

# LANGKAH 3: Tentukan folder kerja di dalam server
WORKDIR /app

# LANGKAH 4: Salin file requirements.txt dan install semua library Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# LANGKAH 5: Salin semua sisa kode aplikasi Anda (termasuk folder api/)
COPY . .

# LANGKAH 6: Perintah untuk menjalankan aplikasi saat server dimulai
# Ganti 'api.index:app' jika nama file atau variabel flask Anda berbeda
# $PORT akan disediakan otomatis oleh Vercel
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "api.index:app"]