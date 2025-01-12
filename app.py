# app.py
from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import sys
from main_fe import main
# Import your existing detection code
# sys.path.append('path_to_your_detection_code')
# from main import your_detection_function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/raw'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max-limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_food():
    try:
        if 'file' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Handle uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image = Image.open(filepath)
        else:
            # Handle base64 image from camera
            image_data = request.form['image'].split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            
        # TODO: Replace this with your actual detection code
        result = main(image)
        print(result['nutrition_info'])
        
        # # Dummy result for demonstration
        # result = {
        #     'detected_items': ['Apple', 'Banana'],
        #     'nutrition_info': {
        #     },
        #     'annotated_image': base64.b64encode(open('path_to_annotated_image', 'rb').read()).decode('utf-8')
        # }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)