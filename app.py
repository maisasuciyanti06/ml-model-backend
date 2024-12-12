import json
import os
import tempfile
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from google.cloud import storage
from google.oauth2 import service_account

app = Flask(__name__)

# Setel kredensial Google Cloud dan klien penyimpanan
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../serviceKey.json'
credentials = service_account.Credentials.from_service_account_file(
    '../serviceKey.json'  # Path ke file kunci akun layanan Anda
)
storage_client = storage.Client(credentials=credentials)

# Muat konfigurasi dari file JSON
with open("config.json") as config_file:
    config = json.load(config_file)
bucket_model = storage_client.bucket(config['bucketModel'])
bucket_images = storage_client.bucket(config['bucketBrain'])

# Variabel global
model_url = 'gs://modelbraintumor/brain_tumor_model.h5'
model = None
tumor_classes = ["meningioma", "glioma", "pituitary", "notumor"]

# Fungsi untuk memuat model dari Google Cloud Storage
def load_model_from_gcs(model_url):
    global model
    if model is None:
        blob = bucket_model.blob(model_url.split('/')[-1])  # Ambil nama blob dari URL
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_model_file:
            blob.download_to_filename(temp_model_file.name)  # Simpan model ke file sementara
            model = tf.keras.models.load_model(temp_model_file.name)  # Muat model dari file
        print("Model dimuat dari GCS")
    return model

# Fungsi untuk mendownload gambar dari URL
def download_image_from_url(image_url):
    try:
        # Jika gambar ada di Google Cloud Storage, download gambar tersebut
        if image_url.startswith('gs://'):
            bucket_name = image_url.split('/')[2]
            blob_name = '/'.join(image_url.split('/')[3:])
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            image_data = blob.download_as_bytes()
        else:
            # Jika URL gambar adalah HTTP(S)
            image_data = requests.get(image_url).content
        return image_data
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat mendownload gambar: {str(e)}")

# Fungsi untuk melakukan prediksi klasifikasi
def predict_classification(image_data):
    try:
        # Decode gambar menjadi tensor
        image = tf.io.decode_jpeg(image_data, channels=3)
        image = tf.image.resize(image, [150, 150])
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32) / 255.0

        # Lakukan prediksi menggunakan model
        predictions = model.predict(image)
        scores = predictions[0]  # Ambil skor prediksi untuk batch pertama

        # Temukan indeks skor tertinggi
        max_score_index = np.argmax(scores)
        label = tumor_classes[max_score_index]

        # Hitung akurasi (dalam persen)
        accuracy = float(scores[max_score_index] * 100)

        return {"label": label, "accuracy": accuracy}
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat prediksi: {str(e)}")

# URL root untuk health check
@app.route('/')
def home():
    return jsonify({"message": "Selamat datang di Layanan Prediksi ML!"})

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil ID pasien dan URL gambar dari body request
        patient_id = request.json.get('patientId')
        image_url = request.json.get('imageUrl')

        if not patient_id or not image_url:
            return jsonify({"error": "ID Pasien atau URL Gambar tidak ditemukan."}), 400

        # Download gambar dari URL yang diberikan
        image_data = download_image_from_url(image_url)

        # Muat model jika belum dimuat
        model = load_model_from_gcs(model_url)
        if model is None:
            return jsonify({"error": "Model tidak dapat dimuat."}), 500

        # Lakukan prediksi pada gambar yang didownload
        result = predict_classification(image_data)

        # Kirim hasil prediksi (Anda juga bisa mengirimkan data ini ke backend utama jika diperlukan)
        return jsonify({
            "label": result["label"],
            "accuracy": result["accuracy"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4002))  # Gunakan port dari environment atau default ke 4002
    app.run(host='0.0.0.0', port=port)
