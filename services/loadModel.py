import tensorflow as tf
import os

def load_model():
    # Muat model menggunakan URL dari variabel lingkungan
    model_url = os.getenv("https://storage.googleapis.com/penyimpanan123/model.json")
    if not model_url:
        raise ValueError("MODEL_URL tidak ditemukan di lingkungan.")
    
    return tf.keras.models.load_model(model_url)
