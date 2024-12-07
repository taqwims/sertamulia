from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflowjs as tfjs
import os
import requests
import uuid
from datetime import datetime
from google.cloud import firestore, secretmanager, storage
import json

try:
    from firebase_admin import credentials
    import firebase_admin
except ImportError:
    print("Firebase Admin SDK tidak installed. Data tidak akan disimpan ke Firestore.")

app = Flask(__name__)

# Inisialisasi Secret Manager dan Firebase
try:
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    secret_name = f"projects/{project_id}/secrets/submission/versions/latest"

    response = client.access_secret_version(request={"name": secret_name})
    secret_string = response.payload.data.decode("UTF-8")
    secret_json = json.loads(secret_string)

    if "submission" in secret_json:
        cred = credentials.Certificate(secret_json["submission"])
        firebase_admin.initialize_app(cred)
        db = firestore.Client()
    else:
        print("Kredensial Firebase tidak ditemukan dalam secret.")
        db = None

except Exception as e:
    print(f"Error in initializing Firebase/Secret Manager: {e}")
    db = None

# URL Model TensorFlow.js
MODEL_URL = os.environ.get("MODEL_URL", "https://storage.googleapis.com/")
LOCAL_MODEL_PATH = "/tmp/model.json"

# Fungsi untuk mengunduh dan memuat model
def load_model_from_json(url, local_path):
    try:
        # Unduh model.json
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

        # Simpan model ke Firestore
        save_model_to_firestore(local_path)

        # Muat model menggunakan TensorFlow.js
        model = tfjs.converters.load_keras_model(local_path)
        print("Model berhasil dimuat.")
        return model
    except requests.exceptions.RequestException as e:
        print(f"Error saat mengunduh model: {e}")
    except Exception as e:
        print(f"Error saat memuat model: {e}")
    return None

def save_model_to_firestore(local_path):
    if db:
        try:
            with open(local_path, "r") as f:
                model_json = f.read()
            doc_ref = db.collection("models").document("model_json")
            doc_ref.set({"model": model_json})
            print("Model berhasil disimpan ke Firestore.")
        except Exception as e:
            print(f"Error saat menyimpan model ke Firestore: {e}")

# Muat model saat aplikasi dijalankan
model = load_model_from_json(MODEL_URL, LOCAL_MODEL_PATH)

class ClientError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code
        self.name = "ClientError"

class InputError(ClientError):
    def __init__(self, message):
        super().__init__(message)
        self.name = "InputError"

def store_data(id, data):
    if db:
        try:
            doc_ref = db.collection("predictions").document(id)
            doc_ref.set(data)
        except Exception as e:
            print(f"Error menyimpan data ke Firestore: {e}")

def predict_classification(model, image_bytes):
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
        raise InputError(f"Kesalahan input: {str(e)}")

@app.route("/predict", methods=["POST"])
def post_predict_handler():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"status": "fail", "message": "No image provided"}), 400

    image = request.files["image"]
    image_bytes = image.read()

    try:
        confidence_score, label, suggestion = predict_classification(model, image_bytes)

        id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        data = {
            "id": id,
            "result": label,
            "suggestion": suggestion,
            "createdAt": created_at,
        }

        store_data(id, data)

        message = "Prediction successful." if confidence_score > 50 else "Prediction completed but below confidence threshold."
        return jsonify({"status": "success", "message": message, "data": data}), 201

    except InputError as e:
        return jsonify({"status": "fail", "message": str(e)}), e.status_code
    except Exception as e:
        return jsonify({"status": "error", "message": f"Internal Server Error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))