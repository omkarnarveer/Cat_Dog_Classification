from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('C:/Users/Omkar/OneDrive/Documents/Cat_Dog_Classification/cat_dog_model.h5')
#model = load_model('C:/Users/Omkar/OneDrive/Documents/Cat_Dog_Classification/cat_dog_model.keras')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected", 400
    
    try:
        # Convert FileStorage object to BytesIO
        img = Image.open(file.stream)  # Load the image using PIL
        img = img.resize((64, 64))  # Resize to the model's input size
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict using the model
        prediction = model.predict(img_array)
        result = "Dog" if prediction[0][0] > 0.5 else "Cat"
        return f"The uploaded image is a: {result}"
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)