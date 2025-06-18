import requests
import base64
import os

# Dapatkan path absolut ke folder known_faces relatif ke lokasi file ini
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'data', 'known_faces', '221401057_Han Ji Eun.jpeg')

# Baca file gambar dan encode ke base64
with open(file_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Siapkan data JSON
data = {
    "image": f"data:image/jpeg;base64,{encoded_string}"
}

# Kirim POST request ke server
url = "http://localhost:5000/attendance"
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

print("Status:", response.status_code)
print("Response:", response.json())
