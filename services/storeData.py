from google.cloud import firestore

def store_data(id, data):
    # Inisialisasi Firestore
    db = firestore.Client()

    # Simpan data ke koleksi `prediction`
    predict_collection = db.collection('prediction')
    predict_collection.document(id).set(data)
