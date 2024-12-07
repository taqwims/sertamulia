import os
import uuid
import json
import logging
from datetime import datetime

import tensorflow as tf
import tensorflowjs as tfjs
import requests

from flask import Flask, request, jsonify
from google.cloud import storage, firestore
from google.oauth2 import service_account
from google.cloud.firestore import SERVER_TIMESTAMP

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/app.log')
    ]
)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi dari environment variables
MODEL_URL = os.environ.get("MODEL_URL")
LOCAL_MODEL_PATH = "/penyimpanan123/model.json"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET_NAME = 'penyimpanan123'

# Global variabel
db = None
model = None
storage_client = None

def get_credentials_from_env_var():
    """Mengambil kredensial dari variabel lingkungan"""
    try:
        credentials_json = os.getenv("submission")
        if not credentials_json:
            logging.error("GCP_CREDENTIALS environment variable not set or empty")
            return None
        
        credentials_info = json.loads(credentials_json)
        return service_account.Credentials.from_service_account_info(credentials_info)
    except Exception as e:
        logging.error(f"Error fetching credentials: {e}")
        return None

def initialize_clients():
    """Inisialisasi Google Cloud clients"""
    global db, storage_client
    try:
        credentials = get_credentials_from_env_var()
        if credentials:
            storage_client = storage.Client(credentials=credentials)
            db = firestore.Client(credentials=credentials)
            logging.info("Google Cloud clients berhasil diinisialisasi")
            return True
        return False
    except Exception as e:
        logging.error(f"Error inisialisasi clients: {e}")
        return False

def upload_to_gcs(local_path, gcs_path):
    """Unggah file ke Google Cloud Storage"""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logging.info(f"File {local_path} berhasil diunggah ke {gcs_path}")
        return blob.public_url
    except Exception as e:
        logging.error(f"Error mengunggah ke GCS: {e}")
        raise

def save_metadata_to_firestore(collection_name, document_id, data):
    """Simpan metadata ke Firestore"""
    if db:
        try:
            data['uploaded_at'] = SERVER_TIMESTAMP
            doc_ref = db.collection(collection_name).document(document_id)
            doc_ref.set(data)
            logging.info(f"Dokumen {document_id} berhasil disimpan ke Firestore")
        except Exception as e:
            logging.error(f"Error menyimpan ke Firestore: {e}")
            raise

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
        
        # Upload model ke GCS setelah diunduh
        gcs_path = f"models/{os.path.basename(local_path)}"
        gcs_url = upload_to_gcs(local_path, gcs_path)
        
        # Simpan metadata model
        model_metadata = {
            "model_url": gcs_url,
            "original_url": url,
            "downloaded_at": datetime.now().isoformat()
        }
        save_metadata_to_firestore("models", str(uuid.uuid4()), model_metadata)
        
        logging.info("Model berhasil diunduh dan diunggah ke GCS")
        return True
    
    except Exception as e:
        logging.error(f"Kesalahan download model: {e}")
        return False

def load_model(url):
    """Muat model TensorFlow.js"""
    try:
        logging.info("Memuat model TensorFlow.js")
        model = tfjs.converters.load_keras_model(url)
        logging.info("Model berhasil dimuat")
        return model
    except Exception as e:
        logging.error(f"Gagal memuat model: {e}")
        return None

def predict_classification(model, image_bytes):
    """Lakukan klasifikasi gambar"""
    try:
        image = tf.image.decode_image(image_bytes, channels=3)
        image = tf.image.resize(image, [224, 224])
        input_tensor = tf.expand_dims(image, 0)
        input_tensor = tf.cast(input_tensor, dtype=tf.float32) / 255.0

        predictions = model.predict(input_tensor)
        confidence_score = float(tf.reduce_max(predictions).numpy()) * 100
        class_result = int(tf.argmax(predictions, axis=1).numpy()[0])

        classes = ["Melanocytic nevus", "Squamous cell carcinoma", "Vascular lesion"]
        label = classes[class_result]

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

    if model is None:
        logging.warning("Model belum dimuat")
        return jsonify({"status": "error", "message": "Model tidak tersedia"}), 500

    if "image" not in request.files:
        return jsonify({"status": "gagal", "message": "Tidak ada gambar"}), 400

    image = request.files["image"]
    image_bytes = image.read()

    try:
        confidence_score, label, suggestion = predict_classification(model, image_bytes)

        # Upload gambar ke GCS
        prediction_id = str(uuid.uuid4())
        gcs_path = f"predictions/{prediction_id}.jpg"
        image_url = upload_to_gcs(image_bytes, gcs_path)

        data = {
            "id": prediction_id,
            "result": label,
            "suggestion": suggestion,
            "confidence": confidence_score,
            "createdAt": datetime.now().isoformat()
        }

        # Simpan hasil prediksi ke Firestore
        save_metadata_to_firestore("predictions", prediction_id, data)

        pesan = "Prediksi berhasil" if confidence_score > 90 else "Prediksi diselesaikan dengan kepercayaan rendah"
        return jsonify({
            "status": "sukses", 
            "message": pesan, 
            "data": data
        }), 201

    except Exception as e:
        logging.error(f"Kesalahan prediksi: {e}")
        return jsonify({"status": "error", "message": f"Kesalahan internal: {e}"}), 500

def setup_application():
    """Menyiapkan aplikasi dengan inisialisasi clients dan model"""
    global model
    
    if not MODEL_URL:
        logging.error("MODEL_URL tidak diset")
        return False
    
    # Inisialisasi Google Cloud clients
    if not initialize_clients():
        return False
    
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