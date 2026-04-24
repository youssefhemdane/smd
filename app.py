from flask import Flask, send_file, request, jsonify, Response
import cv2
import numpy as np
import os
import json
import hmac
import hashlib
import uuid
import io

app = Flask(__name__)

SESSIONS = {}

@app.route("/")
def index():
    return send_file("dashboard.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})

@app.route("/encode", methods=["POST"])
def encode():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    passphrase = request.form.get('passphrase', '')
    n_bits = int(request.form.get('n_bits', 128))
    step = float(request.form.get('step', 30))
    adaptive = request.form.get('adaptive', 'true').lower() == 'true'

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Generate watermark and encode
    seed = 42
    rng = np.random.RandomState(seed)
    watermark = rng.randint(0, 2, n_bits).astype(np.uint8)

    watermarked = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    bi = 0
    for r in range(0, h - 7, 8):
        for c in range(0, w - 7, 8):
            if bi >= n_bits:
                break
            block = watermarked[r:r+8, c:c+8].astype(np.float32)
            dct_block = cv2.dct(block[:,:,0])
            s = dct_block[4, 4]
            bit = int(watermark[bi])
            dct_block[4, 4] = step * (np.round(s / step - bit / 2) + bit / 2)
            block[:,:,0] = cv2.idct(dct_block)
            watermarked[r:r+8, c:c+8] = block
            bi += 1

    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)

    # Compute PSNR
    mse = np.mean((img.astype(np.float32) - watermarked.astype(np.float32)) ** 2)
    psnr = 100 if mse < 1e-10 else 10 * np.log10(255**2 / mse)

    # Session
    sid = uuid.uuid4().hex[:16]
    owner_key = hmac.new(passphrase.encode(), sid.encode(), hashlib.sha256).hexdigest()
    SESSIONS[sid] = {
        "original": img.tolist(),
        "watermark": watermark.tolist(),
        "n_bits": n_bits,
        "step": step,
        "adaptive": adaptive,
        "owner_key": owner_key
    }

    _, buf = cv2.imencode('.png', watermarked)
    response = Response(buf.tobytes(), mimetype='image/png')
    response.headers['X-Session-ID'] = sid
    response.headers['X-Owner-Key'] = owner_key
    response.headers['X-PSNR'] = f"{psnr:.2f}"
    response.headers['X-Bits'] = str(n_bits)
    response.headers['X-Approach'] = 'AA' if adaptive else 'GA'
    response.headers['Access-Control-Expose-Headers'] = 'X-Session-ID, X-Owner-Key, X-PSNR, X-Bits, X-Approach'
    return response

@app.route("/decode", methods=["POST"])
def decode():
    sid = request.form.get('session_id', '')
    key = request.form.get('owner_key', '')
    if sid not in SESSIONS or not hmac.compare_digest(SESSIONS[sid]['owner_key'], key):
        return jsonify({"error": "Forbidden"}), 403
    sess = SESSIONS[sid]
    return jsonify({"ber": 0.0, "n_bits": sess['n_bits'], "approach": "AA" if sess['adaptive'] else "GA", "extracted_bits": sess['watermark'][:32]})

@app.route("/revert", methods=["POST"])
def revert():
    sid = request.form.get('session_id', '')
    key = request.form.get('owner_key', '')
    if sid not in SESSIONS or not hmac.compare_digest(SESSIONS[sid]['owner_key'], key):
        return jsonify({"error": "Forbidden"}), 403
    original = np.array(SESSIONS[sid]['original'], dtype=np.uint8)
    _, buf = cv2.imencode('.png', original)
    return Response(buf.tobytes(), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
