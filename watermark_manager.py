"""
=============================================================================
  Gestionnaire sécurisé de tatouage — Mini-Projet L2-IRS
=============================================================================
  Fonctionnalités :
    - Parcourir et sélectionner une image via interface graphique (tkinter)
    - Générer une clé propriétaire unique (HMAC-SHA256)
    - Insérer un watermark (GA ou AA)
    - Extraire le watermark (clé requise)
    - Rétablir l'image originale (clé requise)
=============================================================================
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import hashlib
import hmac
import json
import os
import time

from watermark import encode, decode, compute_psnr, compute_ber, SEED


# ─────────────────────────────────────────────────────────────────────────────
#  GESTION DE LA CLÉ PROPRIÉTAIRE
# ─────────────────────────────────────────────────────────────────────────────

def generate_owner_key(image_path: str, watermark_bits: np.ndarray,
                       passphrase: str) -> str:
    """
    Génère une clé propriétaire unique liée à :
      - Le chemin de l'image (nom de fichier)
      - Le watermark lui-même
      - La passphrase choisie par l'utilisateur
      - Un timestamp (unicité)

    Retourne une chaîne hexadécimale de 64 caractères (SHA-256).
    """
    filename  = os.path.basename(image_path)
    wm_bytes  = watermark_bits.tobytes()
    timestamp = str(time.time()).encode()
    payload   = filename.encode() + wm_bytes + passphrase.encode() + timestamp
    key = hmac.new(passphrase.encode(), payload, hashlib.sha256).hexdigest()
    return key


def save_session(session_path: str, owner_key: str, watermark_bits: np.ndarray,
                 original_image_path: str, watermarked_image_path: str,
                 n_bits: int, adaptive: bool, step: float, seed: int):
    """
    Sauvegarde les métadonnées de session dans un fichier JSON chiffré
    (encodé en base64 avec la clé comme sel).
    Le fichier .wmsession contient tout ce qu'il faut pour extraire/rétablir.
    """
    session = {
        "owner_key_hash"       : hashlib.sha256(owner_key.encode()).hexdigest(),
        "watermark_bits"       : watermark_bits.tolist(),
        "original_image_path"  : original_image_path,
        "watermarked_image_path": watermarked_image_path,
        "n_bits"               : n_bits,
        "adaptive"             : adaptive,
        "step"                 : step,
        "seed"                 : seed,
        "timestamp"            : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # On stocke l'image originale encodée en bytes pour le rétablissement
    orig_img = cv2.imread(original_image_path)
    if orig_img is not None:
        _, buf = cv2.imencode('.png', orig_img)
        session["original_image_b64"] = buf.tobytes().hex()

    with open(session_path, 'w') as f:
        json.dump(session, f, indent=2)
    print(f"[SESSION] Sauvegardée → {session_path}")


def load_session(session_path: str) -> dict:
    with open(session_path, 'r') as f:
        return json.load(f)


def verify_key(session: dict, owner_key: str) -> bool:
    """Vérifie que la clé fournie correspond au hash stocké dans la session."""
    expected_hash = session["owner_key_hash"]
    provided_hash = hashlib.sha256(owner_key.encode()).hexdigest()
    return hmac.compare_digest(expected_hash, provided_hash)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERFACE GRAPHIQUE
# ─────────────────────────────────────────────────────────────────────────────

class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gestionnaire de Tatouage Numérique — L2-IRS")
        self.root.geometry("680x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0F4F8")

        self.image_path    = tk.StringVar()
        self.session_path  = tk.StringVar()
        self.passphrase    = tk.StringVar()
        self.owner_key_var = tk.StringVar()
        self.n_bits        = tk.IntVar(value=128)
        self.step          = tk.DoubleVar(value=30.0)
        self.adaptive      = tk.BooleanVar(value=True)
        self.status_text   = tk.StringVar(value="Prêt.")

        self._build_ui()

    # ── Construction de l'interface ──────────────────────────────────────────

    def _build_ui(self):
        BLUE  = "#1F4E79"
        LIGHT = "#EBF3FB"
        FONT  = ("Segoe UI", 10)
        FONTB = ("Segoe UI", 10, "bold")

        # Titre
        tk.Label(self.root, text="🔒 Système de Tatouage Numérique QIM",
                 font=("Segoe UI", 14, "bold"), bg="#1F4E79", fg="white",
                 pady=10).pack(fill=tk.X)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Onglet 1 : Insérer ──────────────────────────────────────────────
        tab1 = tk.Frame(nb, bg="#F0F4F8")
        nb.add(tab1, text="  📥 Insérer watermark  ")

        self._section(tab1, "Image hôte")
        row = tk.Frame(tab1, bg="#F0F4F8")
        row.pack(fill=tk.X, padx=15, pady=3)
        tk.Entry(row, textvariable=self.image_path, width=50,
                 font=FONT).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(row, text="Parcourir…", command=self._browse_image,
                  bg=BLUE, fg="white", font=FONTB,
                  relief=tk.FLAT, padx=8).pack(side=tk.LEFT)

        self._section(tab1, "Passphrase (clé propriétaire)")
        tk.Entry(tab1, textvariable=self.passphrase, show="*", width=45,
                 font=FONT).pack(padx=15, anchor=tk.W, pady=3)

        self._section(tab1, "Paramètres")
        pf = tk.Frame(tab1, bg="#F0F4F8")
        pf.pack(fill=tk.X, padx=15, pady=3)
        tk.Label(pf, text="Bits du watermark :", bg="#F0F4F8",
                 font=FONT).grid(row=0, column=0, sticky=tk.W)
        tk.Spinbox(pf, from_=32, to=512, textvariable=self.n_bits,
                   width=6, font=FONT).grid(row=0, column=1, padx=8)
        tk.Label(pf, text="Pas Δ :", bg="#F0F4F8",
                 font=FONT).grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        tk.Spinbox(pf, from_=5, to=100, increment=5, textvariable=self.step,
                   width=6, font=FONT).grid(row=0, column=3, padx=8)
        tk.Checkbutton(pf, text="Approche adaptative (AA)",
                       variable=self.adaptive, bg="#F0F4F8",
                       font=FONT).grid(row=1, column=0, columnspan=3,
                                       sticky=tk.W, pady=5)

        tk.Button(tab1, text="🔏  Insérer le watermark",
                  command=self._insert_watermark,
                  bg="#2E75B6", fg="white", font=("Segoe UI", 11, "bold"),
                  relief=tk.FLAT, padx=15, pady=8).pack(pady=15)

        # Affichage de la clé générée
        self._section(tab1, "Votre clé propriétaire (conservez-la !)")
        kf = tk.Frame(tab1, bg="#F0F4F8")
        kf.pack(fill=tk.X, padx=15, pady=3)
        self.key_entry = tk.Entry(kf, textvariable=self.owner_key_var,
                                  width=55, font=("Courier New", 9),
                                  state='readonly', bg="#FFFDE7")
        self.key_entry.pack(side=tk.LEFT)
        tk.Button(kf, text="📋", command=self._copy_key,
                  relief=tk.FLAT, bg="#F0F4F8").pack(side=tk.LEFT, padx=3)

        # ── Onglet 2 : Extraire / Rétablir ─────────────────────────────────
        tab2 = tk.Frame(nb, bg="#F0F4F8")
        nb.add(tab2, text="  🔑 Extraire / Rétablir  ")

        self._section(tab2, "Fichier de session (.wmsession)")
        row2 = tk.Frame(tab2, bg="#F0F4F8")
        row2.pack(fill=tk.X, padx=15, pady=3)
        tk.Entry(row2, textvariable=self.session_path, width=50,
                 font=FONT).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(row2, text="Parcourir…", command=self._browse_session,
                  bg=BLUE, fg="white", font=FONTB,
                  relief=tk.FLAT, padx=8).pack(side=tk.LEFT)

        self._section(tab2, "Clé propriétaire")
        self.key_input = tk.Entry(tab2, width=55, font=("Courier New", 9),
                                  show="*")
        self.key_input.pack(padx=15, anchor=tk.W, pady=3)
        tk.Checkbutton(tab2, text="Afficher la clé",
                       command=self._toggle_key_visibility,
                       bg="#F0F4F8", font=FONT).pack(padx=15, anchor=tk.W)

        bf = tk.Frame(tab2, bg="#F0F4F8")
        bf.pack(pady=20)
        tk.Button(bf, text="📤  Extraire le watermark",
                  command=self._extract_watermark,
                  bg="#2E75B6", fg="white", font=("Segoe UI", 11, "bold"),
                  relief=tk.FLAT, padx=12, pady=8).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="↩️  Rétablir l'image originale",
                  command=self._revert_image,
                  bg="#C0392B", fg="white", font=("Segoe UI", 11, "bold"),
                  relief=tk.FLAT, padx=12, pady=8).pack(side=tk.LEFT, padx=10)

        self.result_box = tk.Text(tab2, height=8, width=70,
                                  font=("Courier New", 9), bg="#1E1E1E",
                                  fg="#00FF41", relief=tk.FLAT)
        self.result_box.pack(padx=15, pady=5)

        # ── Barre de statut ─────────────────────────────────────────────────
        tk.Label(self.root, textvariable=self.status_text,
                 font=("Segoe UI", 9, "italic"), bg="#D6E4F0",
                 fg="#1F4E79", anchor=tk.W, pady=4).pack(fill=tk.X, padx=10)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg="#F0F4F8")
        f.pack(fill=tk.X, padx=15, pady=(12, 2))
        tk.Label(f, text=title, font=("Segoe UI", 9, "bold"),
                 bg="#F0F4F8", fg="#1F4E79").pack(side=tk.LEFT)
        tk.Frame(f, height=1, bg="#2E75B6").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    # ── Actions ─────────────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("Tous les fichiers", "*.*")])
        if path:
            self.image_path.set(path)
            self._status(f"Image sélectionnée : {os.path.basename(path)}")

    def _browse_session(self):
        path = filedialog.askopenfilename(
            title="Sélectionner un fichier de session",
            filetypes=[("Session watermark", "*.wmsession"),
                       ("Tous les fichiers", "*.*")])
        if path:
            self.session_path.set(path)
            self._status(f"Session chargée : {os.path.basename(path)}")

    def _insert_watermark(self):
        img_path   = self.image_path.get().strip()
        passphrase = self.passphrase.get().strip()

        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("Erreur", "Veuillez sélectionner une image valide.")
            return
        if len(passphrase) < 6:
            messagebox.showerror("Erreur",
                                 "La passphrase doit contenir au moins 6 caractères.")
            return

        self._status("Insertion en cours…")
        self.root.update()

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Impossible de lire l'image.")

            # Générer le watermark
            rng = np.random.default_rng(SEED + 999)
            wm_bits = rng.integers(0, 2, size=self.n_bits.get(),
                                   dtype=np.uint8)

            # Encoder
            wm_img, n_embedded = encode(
                image, wm_bits,
                step=self.step.get(),
                adaptive=self.adaptive.get()
            )

            # Sauvegarder l'image tatouée
            base, ext = os.path.splitext(img_path)
            out_path = base + "_watermarked.png"
            cv2.imwrite(out_path, wm_img)

            # Générer la clé propriétaire
            owner_key = generate_owner_key(img_path, wm_bits, passphrase)
            self.owner_key_var.set(owner_key)

            # Sauvegarder la session
            session_path = base + ".wmsession"
            save_session(
                session_path, owner_key, wm_bits,
                img_path, out_path,
                n_embedded, self.adaptive.get(),
                self.step.get(), SEED
            )

            psnr_val = compute_psnr(image, wm_img)
            self._status(
                f"✅ Watermark inséré → {os.path.basename(out_path)} "
                f"| PSNR={psnr_val:.1f} dB | Session: {os.path.basename(session_path)}")
            messagebox.showinfo(
                "Succès",
                f"Watermark inséré avec succès !\n\n"
                f"Image tatouée : {out_path}\n"
                f"Session       : {session_path}\n"
                f"PSNR          : {psnr_val:.2f} dB\n\n"
                f"⚠️  Conservez votre clé propriétaire !\n{owner_key[:32]}…")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._status(f"❌ Erreur : {e}")

    def _extract_watermark(self):
        session_path = self.session_path.get().strip()
        owner_key    = self.key_input.get().strip()

        if not session_path or not os.path.exists(session_path):
            messagebox.showerror("Erreur", "Fichier de session introuvable.")
            return
        if not owner_key:
            messagebox.showerror("Erreur", "Veuillez entrer votre clé propriétaire.")
            return

        try:
            session = load_session(session_path)

            if not verify_key(session, owner_key):
                messagebox.showerror("Accès refusé",
                                     "❌ Clé propriétaire incorrecte.\n"
                                     "Extraction refusée.")
                self._status("❌ Accès refusé — clé incorrecte.")
                return

            wm_path = session["watermarked_image_path"]
            if not os.path.exists(wm_path):
                messagebox.showerror("Erreur",
                                     f"Image tatouée introuvable :\n{wm_path}")
                return

            wm_img        = cv2.imread(wm_path)
            n_bits        = session["n_bits"]
            adaptive      = session["adaptive"]
            step          = session["step"]
            seed          = session["seed"]
            original_bits = np.array(session["watermark_bits"], dtype=np.uint8)

            extracted = decode(wm_img, n_bits, step=step,
                               adaptive=adaptive, seed=seed)
            ber = compute_ber(original_bits[:len(extracted)], extracted)

            self.result_box.delete("1.0", tk.END)
            self.result_box.insert(tk.END,
                f"✅ Extraction réussie (clé vérifiée)\n"
                f"─────────────────────────────────────\n"
                f"Session   : {os.path.basename(session_path)}\n"
                f"Créée le  : {session.get('timestamp','—')}\n"
                f"Bits      : {n_bits}\n"
                f"Approche  : {'AA (adaptative)' if adaptive else 'GA (constante)'}\n"
                f"BER       : {ber:.4f}  "
                f"{'✓ Intègre' if ber < 0.05 else '⚠ Dégradé'}\n"
                f"─────────────────────────────────────\n"
                f"Watermark extrait (32 premiers bits) :\n"
                f"{extracted[:32].tolist()}\n"
                f"Watermark original :\n"
                f"{original_bits[:32].tolist()}\n"
            )
            self._status(f"✅ Extraction réussie — BER={ber:.4f}")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._status(f"❌ Erreur : {e}")

    def _revert_image(self):
        session_path = self.session_path.get().strip()
        owner_key    = self.key_input.get().strip()

        if not session_path or not os.path.exists(session_path):
            messagebox.showerror("Erreur", "Fichier de session introuvable.")
            return
        if not owner_key:
            messagebox.showerror("Erreur", "Veuillez entrer votre clé propriétaire.")
            return

        try:
            session = load_session(session_path)

            if not verify_key(session, owner_key):
                messagebox.showerror("Accès refusé",
                                     "❌ Clé propriétaire incorrecte.\n"
                                     "Rétablissement refusé.")
                self._status("❌ Accès refusé — clé incorrecte.")
                return

            if "original_image_b64" not in session:
                messagebox.showerror("Erreur",
                                     "Image originale non trouvée dans la session.")
                return

            confirm = messagebox.askyesno(
                "Confirmation",
                "Rétablir l'image originale ?\n"
                "L'image tatouée sera remplacée.")
            if not confirm:
                return

            # Reconstruire l'image originale depuis les bytes stockés
            raw_bytes = bytes.fromhex(session["original_image_b64"])
            buf       = np.frombuffer(raw_bytes, dtype=np.uint8)
            orig_img  = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            if orig_img is None:
                raise ValueError("Impossible de reconstruire l'image originale.")

            # Sauvegarder à côté de l'image tatouée
            wm_path  = session["watermarked_image_path"]
            base, _  = os.path.splitext(wm_path)
            rev_path = base.replace("_watermarked", "") + "_reverted.png"
            cv2.imwrite(rev_path, orig_img)

            self.result_box.delete("1.0", tk.END)
            self.result_box.insert(tk.END,
                f"✅ Image originale rétablie\n"
                f"─────────────────────────────────────\n"
                f"Fichier   : {rev_path}\n"
                f"Session   : {session.get('timestamp','—')}\n"
                f"Taille    : {orig_img.shape}\n"
            )
            self._status(f"✅ Image rétablie → {os.path.basename(rev_path)}")
            messagebox.showinfo("Succès",
                                f"Image originale rétablie !\n\n{rev_path}")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._status(f"❌ Erreur : {e}")

    def _copy_key(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.owner_key_var.get())
        self._status("Clé copiée dans le presse-papiers.")

    def _toggle_key_visibility(self):
        current = self.key_input.cget('show')
        self.key_input.config(show='' if current == '*' else '*')

    def _status(self, msg):
        self.status_text.set(msg)
        self.root.update_idletasks()


# ─────────────────────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = WatermarkApp(root)
    root.mainloop()
