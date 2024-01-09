# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

# Import necessary libraries
import cv2
import numpy as np
from darknet import *
from google.colab import files
from IPython.display import HTML, display
import io
from base64 import b64encode

# Function to perform object tracking in a video
def track_objects(video_path):
    # Load YOLOv4 model
    network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
    width = network_width(network)
    height = network_height(network)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video details
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an output video file
    output_path = '/content/output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the video
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

        # Loop through detections and draw bounding boxes on the frame
        person_detected = False  # Flag to check if person is detected
        for label, confidence, bbox in detections:
            if label == "person":
                left, top, right, bottom = bbox2points(bbox)
                left, top, right, bottom = (
                    int(left * width_ratio),
                    int(top * height_ratio),
                    int(right * width_ratio),
                    int(bottom * height_ratio),
                )

                # Check if the person is in the top or bottom half
                middle_y = (top + bottom) // 2
                if middle_y < height // 2:
                    text_color = (0, 0, 255)  # Red color for top half

                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 4)
                    cv2.putText(
                        frame,
                        "Bus-A",
                        (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        text_color,
                        4,
                    )

                else:
                    text_color = (0, 0, 255)  # Black color for bottom half

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
                    cv2.putText(
                        frame,
                        "Bus-C",
                        (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        text_color,
                        4,
                    )

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()

    print("Object tracking completed. Output video saved as '/content/output.mp4'")
    return output_path

# Provide the specific path for the input video
video_path = '/content/isite_yolo.mp4'

# Perform object tracking in the video
output_path = track_objects(video_path)

# Display the output video
video_file = open(output_path, "rb").read()
video_data = b64encode(video_file).decode("utf-8")
HTML(
    f"""
<video width="640" height="480" controls>
  <source src="data:video/mp4;base64,{video_data}" type="video/mp4">
</video>
"""
)
