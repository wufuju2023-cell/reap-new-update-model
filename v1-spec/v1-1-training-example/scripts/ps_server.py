from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/retrieve_premises", methods=["POST"])
def retrieve():
    data = request.get_json(force=True)
    return jsonify([])


@app.route("/", methods=["POST"])
def gen():
    return jsonify([])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, threaded=True)
