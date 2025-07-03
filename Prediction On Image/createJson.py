import json
import os

# Create a human-readable and understandable class_name
class_name = {
    "c0": "SAFE_DRIVING",
    "c1": "TEXTING_RIGHT",
    "c2": "TALKING_PHONE_RIGHT",
    "c3": "TEXTING_LEFT",
    "c4": "TALKING_PHONE_LEFT",
    "c5": "OPERATING_RADIO",
    "c6": "DRINKING",
    "c7": "REACHING_BEHIND",
    "c8": "HAIR_AND_MAKEUP",
    "c9": "TALKING_TO_PASSENGER"
}

# Define the path where the JSON file will be saved
json_path = r"E:\Distracted-Driver-Detection-Using-Deep-Learning-main\Prediction On Image\class_name_map.json"

# Save the dictionary to the specified JSON file
with open(json_path, 'w') as secret_input:
    json.dump(class_name, secret_input, indent=4, sort_keys=True)
