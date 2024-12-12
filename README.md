**# Cat and Dog Classification Project**

**## Overview**
This project is a machine learning-based image classification system that differentiates between images of cats and dogs. It is built using Python, TensorFlow, and Flask for a user-friendly interface that allows real-time classification of images.

**The system can handle:**
- Image input via the web interface.
- Batch classification using images stored in a directory.

**## Features**
- **Image Upload**: Upload a single image and classify it as a cat or a dog.
- **Pre-trained Model**: Uses a Convolutional Neural Network (CNN) for classification.
- **Real-time Visualization**: Displays the uploaded image and the classification result on the web interface.

**## Prerequisites**
- Python 3.8 or later
- TensorFlow 2.x
- OpenCV
- Flask

**## Project Structure**

Cat_Dog_Classification/
│
├── app.py                     # Flask web application
├── train.py                   # Script to train the model
├── utils.py                   # Helper functions for loading and preprocessing data
├── models/
│   └── cat_dog_model.h5       # Trained model
├── static/
│   ├── css/
│   │   └── style.css          # Stylesheet for the web interface
│   └── uploads/               # Directory to store uploaded images
├── templates/
│   ├── index.html             # Home page for the web app
│   └── result.html            # Result display page
├── data/
│   ├── images/                # Dataset of cat and dog images
│   └── labels.csv             # Labels for training images
├── requirements.txt           # List of required Python packages
└── README.md                  # Project documentation

**##Setup Instructions**
- Clone the Repository:
- git clone https://github.com/omkarnarveer/Cat_Dog_Classification.git
- cd Cat_Dog_Classification
- Install Dependencies: Ensure you have Python installed, then install the required libraries:
- pip install -r requirements.txt
- Train the Model (Optional): If you want to train the model from scratch:
- python train.py
- Run the Application: Start the Flask app:
- python app.py
