from flask import Flask, jsonify
from routes import routes
from services.load_model import load_model
from exceptions.input_error import InputError

app = Flask(__name__)

# Load model and store in app config
app.config['MODEL'] = load_model()

# Register blueprint for routes
app.register_blueprint(routes)

# Error handling
@app.errorhandler(InputError)
def handle_input_error(e):
    response = jsonify({
        "status": "fail",
        "message": f"{e.message} Silakan gunakan foto lain."
    })
    response.status_code = e.status_code
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    response = jsonify({
        "status": "fail",
        "message": str(e)
    })
    response.status_code = 500
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
