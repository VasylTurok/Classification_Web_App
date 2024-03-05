from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

model = load_model("CIFAR_model")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    """Function to check if a file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Define routes"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No selected file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join("uploads", filename))
        img_path = os.path.join("uploads", filename)

        img = image.load_img(img_path, target_size=(32, 32))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        prediction = model.predict(img)
        class_idx = np.argmax(prediction)

        class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        result = class_labels[class_idx]

        os.remove(img_path)

        return render_template("result.html", prediction=result)
    else:
        return render_template("index.html", prediction="Invalid file format")


if __name__ == "__main__":
    app.run(debug=True)
