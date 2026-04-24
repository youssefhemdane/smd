from flask import Flask, send_file

app = Flask(_name_)

@app.route("/")
def index():
    return send_file("dashboard.html")

if _name_ == "__main__":
    app.run(host="0.0.0.0", port=8000)
