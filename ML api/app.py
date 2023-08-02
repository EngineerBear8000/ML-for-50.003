from flask import Flask, redirect, url_for, request
from io import BytesIO
from PIL import Image
from ocr_all_2 import run_ocr
app = Flask(__name__)


@app.route('/', methods=['POST'])
def hello():
    data = request.get_json()
    image = Image.open(BytesIO(bytes(data['file']['value']['data'])))
    type = data['file']['type']
    image.show()

    return run_ocr(image, type)
