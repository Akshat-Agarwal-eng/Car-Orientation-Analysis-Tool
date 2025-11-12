🚗 Car Orientation Analysis Tool@PDAI  

This project is a Python-based computer vision tool designed to analyze images of cars to detect them and determine their orientation (front or rear). It processes a list of image URLs, downloads each image, and runs a multi-stage analysis pipeline.

The final results, including embedded images and detailed JSON data for each car detected, are compiled into an Excel (.xlsx) report.

✨ Key Features

Car Detection: Uses a YOLOv11 (Ultralytics) model to identify and draw bounding boxes around cars in each image.

Multi-Method Orientation: Determines car orientation using a consensus of three different methods for improved accuracy:

Deep Learning Classifier: A fine-tuned MobileNetV2 model classifies cropped car images as "front" or "rear".

Feature-Based Heuristics: A classic CV method that analyzes the lower portion of the car for red pixels (tail lights) or bright pixels (headlights).

Simulated Motion Tracking: A simple tracker assigns IDs to cars and estimates motion direction (simulated per-image) to infer orientation.

Batch Processing: Reads a list of image URLs from a .txt file for automated processing.

Excel Reporting: Generates a clean, professional .xlsx report with embedded (thumbnail) images in one column and the corresponding JSON analysis data in the next.

🛠️ Technologies & Libraries

This script relies on several key Python libraries:

Computer Vision:

ultralytics: For the YOLO car detection model.

opencv-python (cv2): For image loading, processing, and feature analysis.

Pillow (PIL): For image handling and preparation for the Excel report.

Deep Learning:

torch: For loading and running the MobileNetV2 orientation classifier.

torchvision: For image transformations required by the classifier.

Data & Reporting:

numpy: For numerical operations and array manipulation.

openpyxl: For creating and styling the final Excel (.xlsx) report.

Networking:

requests: For downloading images from URLs.

Standard Libraries:

json: For formatting the output results.

collections.defaultdict: For managing car tracks.

🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

1. Prerequisites

You must have Python 3.8 or newer installed. You will also need to install the required libraries.

2. Installation

. Installation

Clone the repository:

git clone [https://github.com/Akshat-Agarwal-eng/Car-Orientation-Analysis-Tool.git](https://github.com/Akshat-Agarwal-eng/Car-Orientation-Analysis-Tool.git)


Navigate to the project directory:

cd Car-Orientation-Analysis-Tool


Install the required Python packages:

pip install -r requirements.txt


If a requirements.txt file is not available, install the libraries manually:

pip install opencv-python-headless numpy ultralytics torch torchvision requests openpyxl pillow



Obtain Model Weights:

YOLO: The script automatically uses yolo11n.pt. This may be a custom name; ensure the correct YOLO weights file (e.g., yolov8n.pt) is available or correctly named in the script.

MobileNetV2 (Orientation): The script is set up to load pre-trained ImageNet weights.

🔔 IMPORTANT: The script includes a commented-out line to load fine-tuned weights:

# orientation_model.load_state_dict(torch.load('car_orientation.pth', map_location=device))
