# Face Recognition Project

A Python-based face recognition application with a GUI that allows users to collect face datasets and perform live face detection using both ANN and CNN models.

## Features

- Interactive GUI for face dataset collection
- Live face detection
- Support for both ANN and CNN face recognition models
- Model comparison functionality
- No command-line arguments required - runs with a single command

## Requirements

- Python 3.10
- OpenCV
- TensorFlow
- scikit-learn
- Pillow (PIL)

All dependencies are listed in `requirements.txt`.

## Setup Instructions

1. Make sure Python 3.10 is installed on your system
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Simply run:
```
python main.py
```

No additional command-line arguments are needed.

## Usage Guide

1. **Start Camera**: Click the "Start Camera" button to activate your webcam.

2. **Dataset Collection**:
   - Enter a person's name in the "Person Name" field
   - Click "Start Collection" to begin capturing face images
   - The system will automatically collect up to 100 samples
   - Click "Stop Collection" to end the process early

3. **Model Training**:
   - Select either "ANN" or "CNN" model type
   - Click "Train Model" to train the selected model on your dataset
   - Wait for training to complete

4. **Face Recognition**:
   - Click "Start Recognition" to begin live face detection
   - The system will display recognized names and confidence levels
   - Click "Stop Recognition" to end the process

## Model Comparison

The application supports two types of face recognition models:

- **ANN (Artificial Neural Network)**: A simpler model that works well for basic face recognition tasks.
- **CNN (Convolutional Neural Network)**: A more advanced model that typically provides better accuracy for face recognition.

You can train and test both models to compare their performance on your specific dataset.

## Project Structure

- `main.py`: Main application file
- `requirements.txt`: List of required Python packages
- `setup.bat`: Windows setup script
- `commands`: Setup instructions for new systems
- `dataset/`: Directory for storing collected face images
- `models/`: Directory for storing trained models
