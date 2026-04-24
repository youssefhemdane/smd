"""
=============================================================================
  API REST — Système de Tatouage Numérique QIM | Mini-Projet L2-IRS
=============================================================================
  Endpoints :
    GET  /                → Dashboard HTML
    GET  /health          → Statut de l'API
    POST /encode          → Insérer un watermark
    POST /decode          → Extraire le watermark (clé requise)
    POST /revert          → Rétablir l'image originale (clé requise)
    POST /attack/jpeg     → Attaque compression JPEG
    POST /attack/gaussian → Attaque bruit gaussien
    POST /metrics         → Calculer PSNR / DWR / BER

  Usage :
    python api.py   →   http://localhost:5000
=============================================================================
"""

from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request
import numpy as np
import cv2
import hashlib, hmac, json, os, io, time

from watermark import (
    encode, decode,
    attack_jpeg, attack_gaussian_noise,
    compute_ber, compute_psnr, compute_dwr,
    SEED
)
from watermark_manager import generate_owner_key, verify_key

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(PROJECT_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Headers custom que le navigateur doit pouvoir lire
EXPOSED_HEADERS = [
    "X-Session-ID", "X-Owner-Key", "X-PSNR",
    "X-DWR", "X-Bits", "X-Approach",
    "Content-Disposition"
]


# ─────────────────────────────────────────────────────────────────────────────
#  CORS — appliqué sur toutes les réponses
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    """
    Ajoute les headers CORS nécessaires à toutes les réponses.
    Sans Access-Control-Expose-Headers, le navigateur bloque la lecture
    des headers personnalisés (X-Session-ID, X-Owner-Key, etc.)
    même quand la requête vient du même hôte.
    """
    response.headers["Access-Control-Allow-Origin"]   = "*"
    response.headers["Access-Control-Allow-Methods"]  = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"]  = "Content-Type"
    response.headers["Access-Control-Expose-Headers"] = ", ".join(EXPOSED_HEADERS)
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    """Répond aux preflight OPTIONS requests."""
    return "", 204


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def _read_image(field="image"):
    """Lit une image uploadée depuis la requête multipart/form-data."""
    if field not in request.files:
        return None, jsonify({"error": f"Champ '{field}' manquant dans la requête."}), 400
    file = request.files[field]
    if file.filename == "":
        return None, jsonify({"error": f"Aucun fichier sélectionné pour '{field}'."}), 400
    data = file.read()
    if len(data) == 0:
        return None, jsonify({"error": f"Le fichier '{field}' est vide."}), 400
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None, jsonify({"error": f"Impossible de décoder l'image '{field}'. Format non supporté."}), 400
    return img, None, 200


def _img_to_response(img, filename="result.png"):
    """Encode une image OpenCV en PNG et la retourne comme réponse HTTP."""
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        return jsonify({"error": "Erreur lors de l'encodage PNG."}), 500
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name=filename
    )


def _parse_bool(val, default=True):
    if val is None:
        return default
    return str(val).strip().lower() in ("true", "1", "yes", "oui")


def _make_wm_bits(n_bits):
    """Génère un watermark binaire pseudo-aléatoire reproductible."""
    rng = np.random.default_rng(SEED + 999)
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def _session_path(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.wmsession")


def _load_session(session_id):
    sf = _session_path(session_id)
    if not os.path.exists(sf):
        return None
    with open(sf, "r") as f:
        return json.load(f)


def _save_session(session_id, data):
    with open(_session_path(session_id), "w") as f:
        json.dump(data, f)


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():
    return send_from_directory(PROJECT_DIR, "dashboard.html")


# ─────────────────────────────────────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "ok",
        "service": "Tatouage QIM API",
        "version": "1.0.0",
        "time"   : time.strftime("%Y-%m-%d %H:%M:%S")
    })


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODE — POST /encode
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/encode", methods=["POST"])
def api_encode():
    """
    Insère un watermark QIM dans l'image.

    Form-data :
      image      (file)   — Image hôte PNG/JPG
      passphrase (str)    — Passphrase ≥ 6 caractères
      n_bits     (int)    — Nombre de bits (défaut: 128)
      step       (float)  — Pas Δ (défaut: 30.0)
      adaptive   (bool)   — AA=true / GA=false (défaut: true)

    Retourne :
      PNG de l'image tatouée avec ces headers :
        X-Session-ID  — ID de session (à conserver)
        X-Owner-Key   — Clé propriétaire (à conserver)
        X-PSNR        — PSNR en dB
        X-DWR         — DWR en dB
        X-Bits        — Nombre de bits insérés
        X-Approach    — AA ou GA
    """
    # 1. Lire l'image
    img, err, code = _read_image("image")
    if err:
        return err, code

    # 2. Valider la passphrase
    passphrase = request.form.get("passphrase", "").strip()
    if len(passphrase) < 6:
        return jsonify({"error": "La passphrase doit faire au moins 6 caractères."}), 400

    # 3. Paramètres
    n_bits   = max(8, int(request.form.get("n_bits", 128)))
    step     = float(request.form.get("step", 30.0))
    adaptive = _parse_bool(request.form.get("adaptive"), default=True)

    # 4. Générer le watermark
    wm_bits = _make_wm_bits(n_bits)

    # 5. Encoder
    try:
        wm_img, n_embedded = encode(img, wm_bits, step=step, adaptive=adaptive)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'encodage : {str(e)}"}), 500

    # 6. Métriques
    try:
        psnr_val = round(float(compute_psnr(img, wm_img)), 2)
        dwr_val  = round(float(compute_dwr(img, wm_img)), 2)
    except Exception:
        psnr_val, dwr_val = 0.0, 0.0

    # 7. Générer session_id et owner_key
    session_id = hashlib.sha256(
        (passphrase + str(time.time()) + str(n_embedded)).encode()
    ).hexdigest()[:16]

    owner_key = generate_owner_key("api_upload", wm_bits, passphrase)

    # 8. Stocker la session (avec image originale pour revert)
    try:
        ok, orig_png = cv2.imencode('.png', img)
        orig_hex     = orig_png.tobytes().hex() if ok else ""
    except Exception:
        orig_hex = ""

    _save_session(session_id, {
        "owner_key_hash"    : hashlib.sha256(owner_key.encode()).hexdigest(),
        "watermark_bits"    : wm_bits.tolist(),
        "n_bits"            : n_embedded,
        "adaptive"          : adaptive,
        "step"              : step,
        "seed"              : SEED,
        "timestamp"         : time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_image_b64": orig_hex,
    })

    # 9. Encoder l'image tatouée en PNG pour la réponse
    ok, out_buf = cv2.imencode('.png', wm_img)
    if not ok:
        return jsonify({"error": "Impossible d'encoder l'image tatouée en PNG."}), 500

    # 10. Construire la réponse avec les headers exposés
    response = send_file(
        io.BytesIO(out_buf.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name="watermarked.png"
    )
    response.headers["X-Session-ID"] = session_id
    response.headers["X-Owner-Key"]  = owner_key
    response.headers["X-PSNR"]       = str(psnr_val)
    response.headers["X-DWR"]        = str(dwr_val)
    response.headers["X-Bits"]       = str(n_embedded)
    response.headers["X-Approach"]   = "AA" if adaptive else "GA"
    return response


# ─────────────────────────────────────────────────────────────────────────────
#  DECODE — POST /decode
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/decode", methods=["POST"])
def api_decode():
    """
    Extrait le watermark d'une image tatouée.

    Form-data :
      image      (file) — Image tatouée (éventuellement attaquée)
      session_id (str)  — ID de session
      owner_key  (str)  — Clé propriétaire

    Retourne : JSON avec bits extraits, BER, intégrité.
    """
    img, err, code = _read_image("image")
    if err:
        return err, code

    session_id = request.form.get("session_id", "").strip()
    owner_key  = request.form.get("owner_key",  "").strip()

    if not session_id or not owner_key:
        return jsonify({"error": "session_id et owner_key sont requis."}), 400

    session = _load_session(session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' introuvable. A-t-elle bien été créée ?"}), 404

    if not verify_key(session, owner_key):
        return jsonify({"error": "Clé propriétaire incorrecte. Accès refusé."}), 403

    try:
        extracted     = decode(img, session["n_bits"],
                               step=session["step"],
                               adaptive=session["adaptive"],
                               seed=session["seed"])
        original_bits = np.array(session["watermark_bits"], dtype=np.uint8)
        ber = float(compute_ber(original_bits[:len(extracted)], extracted))

        return jsonify({
            "status"        : "success",
            "session_id"    : session_id,
            "n_bits"        : len(extracted),
            "ber"           : round(ber, 4),
            "integrity"     : "ok" if ber < 0.05 else "degraded",
            "approach"      : "AA" if session["adaptive"] else "GA",
            "extracted_bits": extracted[:32].tolist(),
            "timestamp"     : session.get("timestamp"),
        })
    except Exception as e:
        return jsonify({"error": f"Erreur de décodage : {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  REVERT — POST /revert
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/revert", methods=["POST"])
def api_revert():
    """
    Rétablit l'image originale depuis la session.

    Form-data :
      session_id (str) — ID de session
      owner_key  (str) — Clé propriétaire

    Retourne : PNG de l'image originale reconstruite.
    """
    session_id = request.form.get("session_id", "").strip()
    owner_key  = request.form.get("owner_key",  "").strip()

    if not session_id or not owner_key:
        return jsonify({"error": "session_id et owner_key sont requis."}), 400

    session = _load_session(session_id)
    if not session:
        return jsonify({"error": f"Session '{session_id}' introuvable."}), 404

    if not verify_key(session, owner_key):
        return jsonify({"error": "Clé propriétaire incorrecte. Accès refusé."}), 403

    if not session.get("original_image_b64"):
        return jsonify({"error": "Image originale non disponible dans cette session."}), 404

    try:
        raw  = bytes.fromhex(session["original_image_b64"])
        buf  = np.frombuffer(raw, dtype=np.uint8)
        orig = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if orig is None:
            raise ValueError("Impossible de reconstruire l'image originale.")
        return _img_to_response(orig, "original_reverted.png")
    except Exception as e:
        return jsonify({"error": f"Erreur de rétablissement : {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  ATTACK JPEG — POST /attack/jpeg
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/attack/jpeg", methods=["POST"])
def api_attack_jpeg():
    img, err, code = _read_image("image")
    if err:
        return err, code
    quality  = max(0, min(100, int(request.form.get("quality", 80))))
    attacked = attack_jpeg(img, quality)
    return _img_to_response(attacked, f"attacked_jpeg_q{quality}.png")


# ─────────────────────────────────────────────────────────────────────────────
#  ATTACK GAUSSIAN — POST /attack/gaussian
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/attack/gaussian", methods=["POST"])
def api_attack_gaussian():
    img, err, code = _read_image("image")
    if err:
        return err, code
    sigma    = float(request.form.get("sigma", 10.0))
    attacked = attack_gaussian_noise(img, sigma)
    return _img_to_response(attacked, f"attacked_gaussian_s{int(sigma)}.png")


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS — POST /metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/metrics", methods=["POST"])
def api_metrics():
    orig, err, code = _read_image("original")
    if err:
        return err, code
    wm, err, code = _read_image("watermarked")
    if err:
        return err, code

    if orig.shape != wm.shape:
        return jsonify({"error": "Les deux images doivent avoir la même résolution."}), 400

    result = {
        "psnr"     : round(float(compute_psnr(orig, wm)), 4),
        "dwr"      : round(float(compute_dwr(orig, wm)),  4),
        "ber"      : None,
        "integrity": None
    }

    session_id = request.form.get("session_id", "").strip()
    owner_key  = request.form.get("owner_key",  "").strip()

    if session_id and owner_key:
        session = _load_session(session_id)
        if session and verify_key(session, owner_key):
            try:
                extracted     = decode(wm, session["n_bits"],
                                       step=session["step"],
                                       adaptive=session["adaptive"],
                                       seed=session["seed"])
                original_bits = np.array(session["watermark_bits"], dtype=np.uint8)
                ber = float(compute_ber(original_bits[:len(extracted)], extracted))
                result["ber"]       = round(ber, 4)
                result["integrity"] = "ok" if ber < 0.05 else "degraded"
            except Exception:
                pass

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  API Tatouage QIM — Mini-Projet L2-IRS")
    print("  http://localhost:5000")
    print("=" * 55)
    print("\n  Endpoints disponibles :")
    print("    GET  /         → Dashboard")
    print("    GET  /health")
    print("    POST /encode")
    print("    POST /decode")
    print("    POST /revert")
    print("    POST /attack/jpeg")
    print("    POST /attack/gaussian")
    print("    POST /metrics")
    print("\n  Sessions stockées dans : ./sessions/")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
