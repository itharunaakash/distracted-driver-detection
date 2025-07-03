# import the necessary packages
from collections import deque
import numpy as np
import cv2
from driver_prediction import predict_result  # Importing your prediction function

# File paths for input and output video
INPUT_VIDEO_FILE = "E:/Distracted-Driver-Detection-Using-Deep-Learning-main/Prediction On Video/input_video.mp4"

OUTPUT_VIDEO_FILE = "output_video.mp4"

# Initialize the video stream
vs = cv2.VideoCapture(INPUT_VIDEO_FILE)
writer = None
(W, H) = (None, None)

# Check if video was successfully opened
if not vs.isOpened():
    print("[ERROR] Unable to open video file.")
    exit()

# Loop over frames from the video file stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # If the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Clone the output frame, convert it from BGR to RGB, resize to 64x64, and preprocess
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    frame = np.expand_dims(frame, axis=0).astype('float32') / 255 - 0.5

    # Make predictions on the frame
    label = predict_result(frame)

    # Display the activity label on the output frame
    text = f"activity: {label}"
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

    # Check if the video writer is None, and initialize it if necessary
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30, (W, H), True)

    # Write the output frame to disk
    writer.write(output)

    # Show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the file pointers
print("[INFO] cleaning up...")
if writer is not None:
    writer.release()
vs.release()
cv2.destroyAllWindows()
