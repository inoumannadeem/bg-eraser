import os
import gdown

def download_u2net():
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'u2net.pth')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(model_path):
        print("Downloading U^2-Net model...")
        url = 'https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab'
        gdown.download(url, model_path, quiet=False)
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")
