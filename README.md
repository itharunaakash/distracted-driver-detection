# Distracted Driver Detection Using Deep Learning

This project uses deep learning to detect distracted driving behaviors from images and videos. It provides:
- A web dashboard (using Dash) for classifying uploaded driver images.
- Scripts for predicting driver behavior in videos.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
  - [Image Prediction Dashboard](#image-prediction-dashboard)
  - [Video Prediction Script](#video-prediction-script)
- [Classes Detected](#classes-detected)
- [Notes](#notes)

---

## Features

- **Image Classification:** Upload a driver image and classify the driver’s activity.
- **Video Classification:** Process a video to label each frame with the predicted driver activity.
- **Pre-trained Model:** Uses a Keras model trained on distracted driver datasets.

---

## Project Structure

```
Distracted-Driver-Detection-Using-Deep-Learning-main/
│
├── Prediction On Image/
│   ├── Dash.py                # Dash web app for image prediction
│   ├── predictionOnImage.py   # Image prediction logic
│   ├── class_name_map.json    # Maps class codes to human-readable names
│   └── ...                    # Other helper scripts
│
├── Prediction On Video/
│   ├── predict_distracted.py  # Video prediction script
│   ├── driver_prediction.py   # Video frame prediction logic
│   └── input_video.mp4        # Example input video
│
├── Training Notebooks/
│   ├── model/
│   │   └── self_trained/
│   │       └── distracted-22-1.00.hdf5  # Trained model (should be present)
│   └── pickle_files/
│       └── labels_list.pkl    # Label mapping for classes
│
├── requirements.txt           # Python dependencies
└── README.md                  # (You are here)
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Distracted-Driver-Detection-Using-Deep-Learning-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model and Data Files:**
   - Ensure the following files exist:
     - `Training Notebooks/model/self_trained/distracted-22-1.00.hdf5` (the trained model)
     - `Training Notebooks/pickle_files/labels_list.pkl`
     - `Prediction On Image/class_name_map.json`
   - If these files are missing, you need to train the model or obtain them from the project author.

---

## How to Use

### Image Prediction Dashboard

1. **Run the Dash app:**
   ```bash
   cd "Prediction On Image"
   python Dash.py
   ```
2. **Open your browser** and go to `http://127.0.0.1:8050/`.
3. **Upload an image** of a driver. Click "Classify" to see the predicted activity.

### Video Prediction Script

1. **Place your input video** in `Prediction On Video/input_video.mp4` (or update the script to use your file).
2. **Run the script:**
   ```bash
   cd "Prediction On Video"
   python predict_distracted.py
   ```
3. The script will process the video, label each frame, and save the output as `output_video.mp4`.

---

## Classes Detected

The model can detect the following driver activities:

- SAFE_DRIVING
- TEXTING_RIGHT
- TALKING_PHONE_RIGHT
- TEXTING_LEFT
- TALKING_PHONE_LEFT
- OPERATING_RADIO
- DRINKING
- REACHING_BEHIND
- HAIR_AND_MAKEUP
- TALKING_TO_PASSENGER

---

## Notes

- The Dash app saves uploaded images as `uploaded_image.jpg` in the project root.
- The model expects images of size 64x64 pixels.
- For best results, use clear images of drivers similar to the training data.
- If you want to retrain the model, refer to the Jupyter notebooks in `Training Notebooks/`.

---
