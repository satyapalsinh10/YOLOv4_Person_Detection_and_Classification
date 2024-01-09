# YOLOv4 - Person Detection and Segmentation

## Introduction
This project demonstrates how to perform object detection using YOLOv4 with the Darknet framework. Implemented YOLO_v4 for efficient person detection and bounding box segmentation in the Queens and Brooklyn MTA bus system, optimizing operations through bus stand segmentation by bus number. Integrated person detection, labeling individuals with their intended bus. By skipping just 1% of stops, saving 39 days annually, the system drastically reduces costs, boosts efficiency, and enhances passenger experience.

## Installation
1. Clone the repository:
   ```
   $ git clone https://github.com/satyapalsinh10/YOLOv4_Person_Detection_and_Segmentation.git
   $ pip install requirements.txt
   ```

2. Run using:
   ```
   python detection.py --video_path /path/to/your/input_video.mp4 --output_path /path/to/your/output_video.mp4
    ```
   
## Citation

  ```
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
  ```

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
