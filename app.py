from flask import Flask, request, send_file, render_template
from rembg import remove
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    
    file = request.files['image']
    if file.filename == '':
        return {'error': 'No image selected'}, 400

    # Read the image
    input_image = Image.open(file.stream)
    
    # Remove background
    output_image = remove(input_image)
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(
        img_byte_arr,
        mimetype='image/png',
        as_attachment=True,
        download_name='removed_bg.png'
    )

if __name__ == '__main__':
    app.run(debug=True)
