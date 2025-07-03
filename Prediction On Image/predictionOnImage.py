import os
import json
from keras.models import load_model
import pickle
import numpy as np
import cv2
from keras.preprocessing import image
from PIL import Image, ImageFile

# Base directory for your model and data
BASE_DIR = "E:\\Distracted-Driver-Detection-Using-Deep-Learning-main\\Training Notebooks"

# Paths for model, pickle files, and JSON files
BASE_MODEL_PATH = os.path.join(BASE_DIR, "model")
PICKLE_DIR = os.path.join(BASE_DIR, "pickle_files")
JSON_DIR = "E:\\Distracted-Driver-Detection-Using-Deep-Learning-main\\Prediction On Image"  # Updated JSON directory

# Load the best model
BEST_MODEL = os.path.join(BASE_MODEL_PATH, "self_trained", "distracted-22-1.00.hdf5")
model = load_model(BEST_MODEL)

# Load labels
with open(os.path.join(PICKLE_DIR, "labels_list.pkl"), "rb") as handle:
    labels_id = pickle.load(handle)

def path_to_tensor(img_path):
    """Load an image from a file path and convert it to a tensor."""
    img = Image.open(img_path)  # Load image from the path
    img = img.resize((64, 64))  # Resize image to match model input
    x = image.img_to_array(img)  # Convert to array
    return np.expand_dims(x, axis=0)  # Expand dims to 4D tensor

def return_prediction(image_path):
    """Return the predicted class for the given image file path."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    test_tensors = path_to_tensor(image_path).astype('float32') / 255 - 0.5

    ypred_test = model.predict(test_tensors, verbose=1)
    
    print(f"Predicted probabilities: {ypred_test}")  # Debugging line
    ypred_class = np.argmax(ypred_test, axis=1)[0]  # Get the predicted class index

    # Create a mapping from indices to class names
    id_labels = {idx: class_name for class_name, idx in labels_id.items()}
    
    print(f"Predicted class index: {ypred_class}")  # Debugging line

    # Get the predicted class name
    res = id_labels.get(ypred_class, "UNKNOWN")  # Use "UNKNOWN" if class not found

    # Load class names from the JSON file
    json_path = os.path.join(JSON_DIR, 'class_name_map.json')  # Corrected path to JSON
    with open(json_path) as secret_input:
        info = json.load(secret_input)

    prediction_result = info.get(res, "UNKNOWN")  # Use "UNKNOWN" if name not found
    return prediction_result

if __name__ == '__main__':
    pass
