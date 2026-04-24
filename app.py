from flask import Flask, send_file, request, jsonify
import subprocess

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("dashboard.html")

@app.route("/encode", methods=["POST"])
def encode():
    # your watermarking logic here
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
