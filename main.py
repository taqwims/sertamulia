import os
import uuid
import json
import logging
from datetime import datetime

import tensorflow as tf
import tensorflowjs as tfjs
import requests

from flask import Flask, request, jsonify
from google.cloud import firestore, secretmanager

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output ke console
        logging.FileHandler('/app/app.log')  # Logging ke file
    ]
)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi dari environment variables
MODEL_URL = os.environ.get("MODEL_URL")
LOCAL_MODEL_PATH = "/tmp/model.json"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

# Global variabel
db = None
model = None

def initialize_firebase():
    """Inisialisasi Firebase dengan Secret Manager"""
    global db
    try:
        # Inisialisasi Secret Manager
        client = secretmanager.SecretManagerServiceClient()
        
        if not PROJECT_ID:
            logging.error("GOOGLE_CLOUD_PROJECT tidak diset")
            return None

        secret_name = f"projects/{PROJECT_ID}/secrets/submission/versions/latest"

        # Akses secret
        response = client.access_secret_version(request={"name": secret_name})
        secret_string = response.payload.data.decode("UTF-8")
        secret_json = json.loads(secret_string)

        # Inisialisasi Firebase Admin
        if "submission" in secret_json:
            from firebase_admin import credentials, initialize_app
            cred = credentials.Certificate(secret_json["submission"])
            initialize_app(cred)
            db = firestore.Client()
            logging.info("Firebase berhasil diinisialisasi")
            return db
        else:
            logging.error("Kredensial Firebase tidak ditemukan dalam secret")
            return None

    except Exception as e:
        logging.error(f"Error inisialisasi Firebase: {e}")
        return None

def download_model(url, local_path):
    """Unduh model dari URL"""
    try:
        logging.info(f"Mencoba mengunduh model dari: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            f.write(response.content)
        
        logging.info("Model berhasil diunduh")
        return True
    
    except Exception as e:
        logging.error(f"Kesalahan download model: {e}")
        return False

def load_model(local_path):
    """Muat model TensorFlow.js"""
    try:
        logging.info("Memuat model TensorFlow.js")
        model = tfjs.converters.load_keras_model(local_path)
        logging.info("Model berhasil dimuat")
        return model
    except Exception as e:
        logging.error(f"Gagal memuat model: {e}")
        return None

def store_prediction_data(data):
    """Simpan data prediksi ke Firestore"""
    global db
    if db:
        try:
            doc_ref = db.collection("predictions").document(data['id'])
            doc_ref.set(data)
            logging.info("Data prediksi berhasil disimpan")
        except Exception as e:
            logging.error(f"Kesalahan menyimpan data prediksi: {e}")

def predict_classification(model, image_bytes):
    """Lakukan klasifikasi gambar"""
    try:
        # Preprocess gambar
        image = tf.image.decode_image(image_bytes, channels=3)
        image = tf.image.resize(image, [224, 224])
        input_tensor = tf.expand_dims(image, 0)
        input_tensor = tf.cast(input_tensor, dtype=tf.float32) / 255.0

        # Prediksi
        predictions = model.predict(input_tensor)
        confidence_score = float(tf.reduce_max(predictions).numpy()) * 100
        class_result = int(tf.argmax(predictions, axis=1).numpy()[0])

        # Label kelas
        classes = ["Melanocytic nevus", "Squamous cell carcinoma", "Vascular lesion"]
        label = classes[class_result]

        # Penjelasan dan saran
        explanation = {
            "Melanocytic nevus": "Kondisi permukaan kulit dengan bercak warna dari sel melanosit.",
            "Squamous cell carcinoma": "Kanker kulit yang sering muncul di area terkena sinar UV.",
            "Vascular lesion": "Tumor atau kanker yang sering muncul di kepala dan leher."
        }.get(label, "Kondisi kulit tidak dikenali")

        suggestion = {
            "Melanocytic nevus": "Konsultasi dokter jika ada perubahan ukuran atau warna.",
            "Squamous cell carcinoma": "Segera konsultasi untuk mencegah penyebaran.",
            "Vascular lesion": "Konsultasikan untuk tindakan lebih lanjut."
        }.get(label, "Disarankan pemeriksaan medis")

        return confidence_score, label, explanation, suggestion

    except Exception as e:
        logging.error(f"Kesalahan prediksi: {e}")
        raise ValueError(f"Gagal melakukan prediksi: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict_handler():
    global model

    # Periksa model
    if model is None:
        logging.warning("Model belum dimuat")
        return jsonify({"status": "error", "message": "Model tidak tersedia"}), 500

    # Periksa keberadaan gambar
    if "image" not in request.files:
        return jsonify({"status": "gagal", "message": "Tidak ada gambar"}), 400

    image = request.files["image"]
    image_bytes = image.read()

    try:
        confidence_score, label, explanation, suggestion = predict_classification(model, image_bytes)

        # Persiapkan data
        data = {
            "id": str(uuid.uuid4()),
            "result": label,
            "explanation": explanation,
            "suggestion": suggestion,
            "confidence": confidence_score,
            "createdAt": datetime.now().isoformat()
        }

        # Simpan data
        store_prediction_data(data)

        # Respon
        pesan = "Prediksi berhasil" if confidence_score > 50 else "Prediksi diselesaikan dengan kepercayaan rendah"
        return jsonify({
            "status": "sukses", 
            "message": pesan, 
            "data": data
        }), 201

    except Exception as e:
        logging.error(f"Kesalahan prediksi: {e}")
        return jsonify({"status": "error", "message": f"Kesalahan internal: {e}"}), 500

def setup_application():
    """Menyiapkan aplikasi dengan inisialisasi Firebase dan model"""
    global model, db
    
    # Validasi URL model
    if not MODEL_URL:
        logging.error("MODEL_URL tidak diset")
        return False
    
    # Inisialisasi Firebase
    db = initialize_firebase()
    
    # Download dan muat model
    if download_model(MODEL_URL, LOCAL_MODEL_PATH):
        model = load_model(LOCAL_MODEL_PATH)
        return model is not None
    
    return False

# Jalankan setup saat aplikasi dimulai
if not setup_application():
    logging.critical("Gagal menyiapkan aplikasi. Periksa konfigurasi.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)