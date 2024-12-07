from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflowjs as tfjs
import os
import requests
import uuid
from datetime import datetime
from google.cloud import firestore, secretmanager, storage
import json
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi URL Model dan Path Lokal
MODEL_URL = os.environ.get("MODEL_URL", "https://storage.googleapis.com/your-bucket/model.json")
LOCAL_MODEL_PATH = "/tmp/model.json"

# Inisialisasi global variabel
db = None
model = None

def initialize_firebase():
    """Inisialisasi Firebase dengan Secret Manager"""
    global db
    try:
        # Inisialisasi Secret Manager
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id:
            logging.error("GOOGLE_CLOUD_PROJECT tidak diset")
            return None

        secret_name = f"projects/{project_id}/secrets/submission/versions/latest"

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
        
        # Tambahkan header untuk memastikan koneksi
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        logging.info(f"Status download: {response.status_code}")
        logging.info(f"Panjang konten: {len(response.content)} bytes")
        
        with open(local_path, "wb") as f:
            f.write(response.content)
        
        logging.info("Model berhasil diunduh")
        return True
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Kesalahan jaringan saat download model: {e}")
        return False
    except IOError as e:
        logging.error(f"Kesalahan IO saat menyimpan model: {e}")
        return False
    except Exception as e:
        logging.error(f"Kesalahan tidak terduga saat download model: {e}")
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

def save_model_to_firestore(local_path):
    """Simpan model ke Firestore"""
    global db
    if db:
        try:
            with open(local_path, "r") as f:
                model_json = f.read()
            
            doc_ref = db.collection("models").document("model_json")
            doc_ref.set({"model": model_json})
            
            logging.info("Model berhasil disimpan ke Firestore")
        except Exception as e:
            logging.error(f"Kesalahan saat menyimpan model ke Firestore: {e}")

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
        explanation, suggestion = "", ""
        if label == "Melanocytic nevus":
            explanation = "Melanocytic nevus adalah kondisi permukaan kulit memiliki bercak warna yang berasal dari sel-sel melanosit."
            suggestion = "Segera konsultasi dengan dokter jika ada perubahan ukuran atau warna."
        elif label == "Squamous cell carcinoma":
            explanation = "Squamous cell carcinoma adalah jenis kanker kulit yang sering muncul di area terkena sinar UV."
            suggestion = "Segera konsultasi dengan dokter untuk mencegah penyebaran."
        elif label == "Vascular lesion":
            explanation = "Vascular lesion adalah tumor atau kanker yang sering muncul di kepala dan leher."
            suggestion = "Konsultasikan dengan dokter untuk tindakan lebih lanjut."

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
        id_prediksi = str(uuid.uuid4())
        waktu_prediksi = datetime.now().isoformat()

        data = {
            "id": id_prediksi,
            "result": label,
            "explanation": explanation,
            "suggestion": suggestion,
            "confidence": confidence_score,
            "createdAt": waktu_prediksi
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
    
    # Inisialisasi Firebase
    db = initialize_firebase()
    
    # Download dan muat model
    if download_model(MODEL_URL, LOCAL_MODEL_PATH):
        model = load_model(LOCAL_MODEL_PATH)
        
        # Simpan model ke Firestore jika berhasil
        if model:
            save_model_to_firestore(LOCAL_MODEL_PATH)
        else:
            logging.error("Gagal memuat model")
    else:
        logging.error("Gagal download model")

# Jalankan setup saat aplikasi dimulai
setup_application()

if __name__ == "__main__":
    app.run(
        debug=True, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8080))
    )