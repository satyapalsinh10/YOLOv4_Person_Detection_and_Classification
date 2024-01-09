# Import necessary libraries
import cv2
import numpy as np
from darknet import *
from IPython.display import HTML, display
import io
from base64 import b64encode
import argparse
from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import PIL
import io
import html
import time
import matplotlib.pyplot as plt

# Function to perform object detection using YOLOv4
def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height

    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)

    return detections, width_ratio, height_ratio

# Function to perform object tracking in a video
def track_objects(video_path, output_path):
    network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
    width = network_width(network)
    height = network_height(network)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

        for label, confidence, bbox in detections:
            if label == "person":
                left, top, right, bottom = bbox2points(bbox)
                left, top, right, bottom = (
                    int(left * width_ratio),
                    int(top * height_ratio),
                    int(right * width_ratio),
                    int(bottom * height_ratio),
                )

                middle_y = (top + bottom) // 2
                if middle_y < height // 2:
                    text_color = (0, 0, 255)

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
                    text_color = (0, 0, 255) 

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

        out.write(frame)

    cap.release()
    out.release()

    print(f"Object tracking completed. Output video saved as '{output_path}'")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object tracking in a video with YOLOv4")

    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_path", required=True, help="Path to the output video file")

    args = parser.parse_args()

    track_objects(args.video_path, args.output_path)
