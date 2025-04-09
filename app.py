from flask import Flask, request, send_file, render_template, jsonify
import requests
import os
from PIL import Image
import io

app = Flask(__name__)

# You'll need to set your Remove.bg API key
API_KEY = 'YOUR-API-KEY'  # Replace with your API key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read the image into memory
        img_byte_arr = io.BytesIO()
        Image.open(file.stream).save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Call remove.bg API
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': img_byte_arr},
            headers={'X-Api-Key': API_KEY},
        )
        
        if response.status_code == 200:
            # Return the processed image
            result = io.BytesIO(response.content)
            result.seek(0)
            return send_file(
                result,
                mimetype='image/png',
                as_attachment=True,
                download_name='removed_bg.png'
            )
        else:
            return jsonify({'error': 'Failed to process image'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
