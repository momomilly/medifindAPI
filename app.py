from flask import Flask, request, jsonify
from detail_model import preprocess_image, predict_image

from PIL import Image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg'}


def allowed_file(filename):
    # xxx.jpg -> it return TRUE if it look like ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return "This is the medical prediction"

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    #check the image file that has null or wrong format
    file = request.files['image']
    if file is None or file.filename == "":
        return jsonify({'error': 'no file'})
    if not allowed_file(file.filename):
        return jsonify({'error': 'format not supported'})

    try:
        #1 load image
        pic = Image.open(file)

        #2 preprocess image
        pre_image = preprocess_image(pic)

        #3 prediction
        result = predict_image(pre_image)

        #4 get the result to json data
        return jsonify({'result': result})

    except:
        return jsonify({'result': "Sorry please predict again"})


if __name__ == '__main__':
    app.run(debug=True)
