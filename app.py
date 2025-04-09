from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
import io
import torch
import os
from u2net import U2NET, remove_background
from download_model import download_u2net

app = Flask(__name__)

# Download and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U2NET()
download_u2net()
model.load_state_dict(torch.load('models/u2net.pth', map_location=device))
model.to(device)
model.eval()

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
        # Read the input image
        input_image = Image.open(file.stream).convert('RGB')
        
        # Process with U^2-Net
        output_image = remove_background(model, input_image)
        
        # Save to bytes
        output_bytes = io.BytesIO()
        output_image.save(output_bytes, format='PNG')
        output_bytes.seek(0)
        
        return send_file(
            output_bytes,
            mimetype='image/png',
            as_attachment=True,
            download_name='removed_bg.png'
        )
    except Exception as e:
        print(f'Error processing image: {str(e)}')
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
