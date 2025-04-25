from flask import Blueprint, request, jsonify, Flask


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the A bunch of Devs!"})