import os
import json
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing import image
from tqdm.notebook import tqdm
from PIL import ImageFile

# Directories
BASE_MODEL_PATH = os.path.join(os.getcwd(), "model")
PICKLE_DIR = "Training Notebooks/pickle_files"
# Path to the class name JSON in "Prediction On Image"
JSON_FILE_PATH = "E:/Distracted-Driver-Detection-Using-Deep-Learning-main/Prediction On Image/class_name_map.json"

# Load the trained model
BEST_MODEL = "Training Notebooks/model/self_trained/distracted-22-1.00.hdf5"
model = load_model(BEST_MODEL)

# Load the label mappings from pickle file
with open(os.path.join(PICKLE_DIR, "labels_list.pkl"), "rb") as handle:
    labels_id = pickle.load(handle)

# Function to preprocess a single image to tensor
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Load image and resize
    x = image.img_to_array(img)                           # Convert to array
    return np.expand_dims(x, axis=0)                      # Add extra dimension for batch

# Function to preprocess multiple image paths to tensors
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated image files

# Prediction function
def predict_result(image_tensor):
    try:
        # Perform prediction
        ypred_test = model.predict(image_tensor, verbose=1)
        ypred_class = np.argmax(ypred_test, axis=1)  # Get the class with highest score
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None
    
    # Reverse the label mapping for easier access
    id_labels = {v: k for k, v in labels_id.items()}
    
    try:
        ypred_class = int(ypred_class)
        print(f"Predicted class index: {ypred_class}")
    except KeyError:
        print(f"Error: Class {ypred_class} not found in label map")
        return None

    # Load the class name map from the correct JSON file path
    with open(JSON_FILE_PATH) as secret_input:
        info = json.load(secret_input)

    try:
        label = info[id_labels[ypred_class]]  # Map predicted class to human-readable label
        print(f"Predicted label: {label}")
    except KeyError:
        print(f"Error: Label for class {id_labels[ypred_class]} not found in JSON")
        label = None
    
    return label
