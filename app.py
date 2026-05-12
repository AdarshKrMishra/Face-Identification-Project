from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    from flask import request
    person_name = request.form.get("name", "Unknown")
    subprocess.run(["python", "core/dataset_creator.py", person_name])
    return render_template("index.html", result=f"Dataset Captured for {person_name} Successfully")


@app.route("/train", methods=["POST"])
def train():
    subprocess.run(["python", "core/train_model.py"])
    return render_template("index.html", result="Model Trained Successfully")


@app.route("/recognize", methods=["POST"])
def recognize():
    subprocess.run(["python", "core/recognize.py"])
    return render_template("index.html", result="Recognition Started (Press ESC to stop camera)")


if __name__ == "__main__":
    app.run(debug=True)