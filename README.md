# Real-Estate-Image-Classifier
Project Overview
This repository contains the implementation of a deep learning model designed to classify images into two distinct categories: property and non-property. The project is structured to provide an end-to-end solution from data preprocessing to model training and serving predictions via a REST API.

Key Features
Data Preprocessing: Includes scripts for preparing and augmenting images to ensure the model receives uniformly preprocessed data, enhancing the model's ability to generalize.
Model Training: Utilizes a Convolutional Neural Network (CNN) to learn distinguishing features between property and non-property images. The model is trained using a dataset categorized into respective folders, ensuring straightforward scalability and manageability.
Flask API: A Flask application serves the trained model, allowing users to submit images in real-time and receive predictions. This component makes the model accessible as a service, ready for integration into larger applications or for standalone use.
Repository Structure


property-image-classifier/
│
├── api/                    # Flask API for serving predictions
│   ├── app.py              # Main script for the Flask application
│   └── requirements.txt    # Dependencies for the API
│
├── model/                  # Scripts for model handling
│   ├── train_model.py      # Script for training the model
│   ├── preprocess.py       # Script for data preprocessing
│   └── model.h5            # Saved model after training
│
├── data/                   # Dataset for training and validation
│   ├── train/              # Training data
│   │   ├── property/       # Images of properties
│   │   └── non_property/   # Images not of properties
│   │
│   └── validation/         # Validation data
│       ├── property/       # Validation images of properties
│       └── non_property/   # Validation images not of properties


Technology Stack
TensorFlow/Keras: For constructing and training the deep learning model.
Flask: To create a REST API for serving the model predictions.
Pillow: For image manipulation during preprocessing.
NumPy: For numerical operations particularly in data handling and transformations.

Usage
1/To train the model:
Navigate to the model directory.
run train_model.py .
2/To run the Flask API locally: 
Navigate to the api directory.
Install dependencies using pip install -r requirements.txt.
Execute python app.py to start the server.
Use a tool like Postman or a similar HTTP client to send POST requests to http://localhost:5000/predict with an image to classify.

Contributing
Contributions to this project are welcome! Please consider forking the repository, making your changes, and submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.
