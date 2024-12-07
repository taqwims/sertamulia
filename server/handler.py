import uuid
from datetime import datetime
from flask import request, jsonify
from services.inference_service import predict_classification
from services.store_data import store_data

def post_predict_handler():
    if 'image' not in request.files:
        return jsonify({
            "status": "fail",
            "message": "File gambar tidak ditemukan dalam permintaan."
        }), 400

    image = request.files['image']
    model = request.app.config['MODEL']

    confidence_score, label, explanation, suggestion = predict_classification(model, image)
    id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    data = {
        "id": id,
        "result": label,
        "explanation": explanation,
        "suggestion": suggestion,
        "confidenceScore": confidence_score,
        "createdAt": created_at
    }

    store_data(id, data)

    response_message = (
        "Model is predicted successfully."
        if confidence_score > 99
        else "Model is predicted successfully but under threshold. Please use the correct picture."
    )

    return jsonify({
        "status": "success",
        "message": response_message,
        "data": data
    }), 201
