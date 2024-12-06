from flask import Flask, request, jsonify
import tensorflow as tf
from google.cloud import storage
import os
import uuid
from datetime import datetime
from google.cloud import firestore, secretmanager
import json
try:
    from firebase_admin import credentials
    import firebase_admin
except ImportError:
    print("Firebase Admin SDK tidak diinstal. Data tidak akan disimpan ke Firestore.")


app = Flask(__name__)

# Inisialisasi Secret Manager dan Firebase
try:
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.environ.get("submissionmlgc-ahsan")
    secret_name = f"projects/{project_id}/secrets/submission/versions/latest" 

    response = client.access_secret_version(request={"name": secret_name})
    secret_string = response.payload.data.decode("UTF-8")
    secret_json = json.loads(secret_string)

    if "submission" in secret_json: # Pastikan key yang benar ada di secret
        cred = credentials.Certificate(secret_json["submission"])
        firebase_admin.initialize_app(cred)
        db = firestore.Client()
    else:
        print("Kredensial Firebase tidak ditemukan dalam secret.")
        db = None

except (Exception, KeyError) as e:
    print(f"Error in initializing Firebase/Secret Manager: {e}")
    db = None


# Load model TensorFlow
MODEL_URL = os.environ.get("penyimpanan123/model.json") # URL model di GCS, contoh: gs://your-bucket/your-model.tf

try:
    gcs_path = MODEL_URL.replace("gs://", "https://storage.googleapis.com/") # Mengubah ke URL HTTP
    model = tf.keras.models.load_model(gcs_path)

except Exception as e:
    print(f"Error loading model from GCS URL: {e}")
    model = None


class ClientError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code
        self.name = 'ClientError'


class InputError(ClientError):
    def __init__(self, message):
        super().__init__(message)
        self.name = 'InputError'


def store_data(id, data):
    if db:
        try:
            doc_ref = db.collection("predictions").document(id)
            doc_ref.set(data)
        except Exception as e:
            print(f"Error menyimpan data ke Firestore: {e}")


def load_model():
    return model


def predict_classification(model, image_bytes):
    try:
        image = tf.image.decode_image(image_bytes, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.expand_dims(image, 0)
        image = tf.cast(image, dtype=tf.float32) / 255.0

        prediction = model.predict(image)
        score = tf.reduce_max(prediction).numpy()
        confidence_score = float(score) * 100

        classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion']
        class_result = tf.argmax(prediction, axis=1).numpy()[0]
        label = classes[class_result]

        explanation, suggestion = "", ""

        if label == 'Melanocytic nevus':
            explanation = "Melanocytic nevus adalah kondisi permukaan kulit memiliki bercak warna yang berasal dari sel-sel melanosit, yakni pembentukan warna kulit dan rambut."
            suggestion = "Segera konsultasi dengan dokter terdekat jika ukuran semakin membesar dengan cepat, mudah luka, atau berdarah."

        elif label == 'Squamous cell carcinoma':
            explanation = "Squamous cell carcinoma adalah jenis kanker kulit yang umum dijumpai. Penyakit ini sering tumbuh pada bagian-bagian tubuh yang sering terkena sinar UV."
            suggestion = "Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker."

        elif label == 'Vascular lesion':
            explanation = "Vascular lesion adalah penyakit yang dikategorikan sebagai kanker atau tumor. Penyakit ini sering muncul pada bagian kepala dan leher."
            suggestion = "Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit."

        return confidence_score, label, explanation, suggestion

    except Exception as e:
        raise InputError(f"Terjadi kesalahan input: {str(e)}")


@app.route('/predict', methods=['POST'])
def post_predict_handler():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'message': 'No image provided'}), 400

    image = request.files['image']
    image_bytes = image.read()

    try:
        confidence_score, label, explanation, suggestion = predict_classification(model, image_bytes)

        id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        data = {
            "id": id,
            "result": label,
            "suggestion": suggestion,
            "createdAt": created_at
        }

        store_data(id, data)

        message = 'Model is predicted successfully.' if confidence_score > 0.99 else 'Model is predicted successfully but under threshold. Please use the correct picture'
        response = jsonify({
            'status': 'success',
            'message': message,
            'data': data
        })
        response.status_code = 201
        return response

    except InputError as e:
        return jsonify({'status': 'fail', 'message': str(e)}), e.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Internal Server Error: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))