from ultralytics import YOLO

# sample video
VIDEO_PATH = "./sample/video3.mp4"

# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track(VIDEO_PATH, show=True, tracker="botsort.yaml", save=True, conf=0.4)  # Tracking with custom tracker