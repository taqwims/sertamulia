import tensorflow as tf
from exceptions.input_error import InputError

def predict_classification(model, image):
    try:
        # Decode gambar dari input dan preprocess
        tensor = (
            tf.image.decode_jpeg(image.read(), channels=3)
            .resize([224, 224])
            .numpy()
        )
        tensor = tf.convert_to_tensor([tensor], dtype=tf.float32)

        # Lakukan prediksi
        prediction = model(tensor)
        scores = prediction.numpy()[0]
        confidence_score = max(scores) * 100

        classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion']
        class_result = int(tf.argmax(prediction, axis=1).numpy()[0])
        label = classes[class_result]

        # Penjelasan dan saran berdasarkan label
        explanation, suggestion = None, None
        if label == 'Melanocytic nevus':
            explanation = "Melanocytic nevus adalah kondisi permukaan kulit memiliki bercak warna yang berasal dari sel-sel melanosit, yakni pembentukan warna kulit dan rambut."
            suggestion = "Segera konsultasi dengan dokter terdekat jika ukuran semakin membesar dengan cepat, mudah luka, atau berdarah."
        elif label == 'Squamous cell carcinoma':
            explanation = "Squamous cell carcinoma adalah jenis kanker kulit yang umum dijumpai. Penyakit ini sering tumbuh pada bagian-bagian tubuh yang sering terkena sinar UV."
            suggestion = "Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker."
        elif label == 'Vascular lesion':
            explanation = "Vascular lesion adalah penyakit yang dikategorikan sebagai kanker atau tumor. Penyakit ini sering muncul pada bagian kepala dan leher."
            suggestion = "Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit."

        return {
            "confidenceScore": confidence_score,
            "label": label,
            "explanation": explanation,
            "suggestion": suggestion
        }
    except Exception as error:
        raise InputError(f"Terjadi kesalahan input: {error}")
