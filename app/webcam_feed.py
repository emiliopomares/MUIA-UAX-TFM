import cv2
import numpy as np
import torch
import base64

from utils.cameras import list_available_webcams

init = False
camera_0 = None
camera_1 = None

# Set desired resolution
frame_width = 1280
frame_height = 720
# Desired output resolution
output_width = 256
output_height = 256

def init_feed():
    global init
    global camera_0
    global camera_1
    if init:
        print("Trying to init again")
        return

    # Get available webcams
    available_webcams = list_available_webcams()
    print(f"[Webcam Feed] available webcams: {available_webcams}")

    # Check if we have at least two cameras
    if len(available_webcams) < 2:
        print(f"[Webcam Feed] exception: at least two cameras are needed, {len(available_webcams)} found")
        import sys
        sys.exit(0)

    # Initialize webcams
    camera_0 = cv2.VideoCapture(available_webcams[0])  # Left webcam
    camera_1 = cv2.VideoCapture(available_webcams[1])  # Right webcam

    camera_0.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera_0.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    camera_1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera_1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    init = True

def convert_image_to_base64(image):
    """Convert a frame (numpy array) into base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

def grab_frame(index, invert=False):
    cam = (camera_0 if index==1 else camera_1) if invert else (camera_0 if index==0 else camera_1)
    success, frame = cam.read()
    if success:
        # Convert to grayscale for simpler statistics
        resized_frame = cv2.resize(frame, (output_width, output_height))
        # Encode frame as JPEG
        #ret, buffer = cv2.imencode('.jpg', resized_frame)
        return resized_frame
    else:
        return None

def make_input_tensor(L, R, invert=False):
    """Makes a (6, 256, 256) input tensor datapoint from L and R images"""
    # Encode frame as JPEG
    L = torch.tensor(L)
    print(f"L shape: {L.shape}")
    L = L.permute(2, 0, 1)
    R = torch.tensor(R)
    print(f"L shape: {R.shape}")
    R = R.permute(2, 0, 1)
    datapoint = torch.cat((L, R), dim=0)
    #print(f"Datapoint shape: {datapoint.shape}")
    return (datapoint.float()/255.0).unsqueeze(0)