from flask import Blueprint
from handler import post_predict_handler

routes = Blueprint('routes', __name__)

@routes.route('/predict', methods=['POST'])
def predict():
    return post_predict_handler()
