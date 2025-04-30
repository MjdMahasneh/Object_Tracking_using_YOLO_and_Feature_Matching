# Object Tracking using YOLO RoIs and Feature Matching

### 🔍 Introduction

While experimenting with **YOLO**, I had the idea to use its **intermediate feature maps** for **similarity analysis** and tracking objects across frames.

This project is a **custom object tracking system** that uses YOLO for detection and combines **feature embedding similarity** with **Kalman filter-based motion prediction** to track objects over time.

- Feature embeddings are extracted from YOLO's internal layers using **ROI Align**.
- Objects are matched across frames using **cosine similarity** and **IoU**, combined in a cost matrix.
- Matching is solved using the **Hungarian algorithm**, and tracks are maintained or terminated based on detection consistency.

> 🛠️ This is a **custom tracker**, inspired by Deep SORT, but built from scratch—no external ReID models or Deep SORT components.

![results](/materials/screenshot.png)

![Object Tracking](/materials/tracking_flowchart.png)



## Object Tracking Process

The object tracking system consists of three main components: **object detection**, **feature extraction**, and **object tracking**.

#### 1. Object Detection  
- The system uses **YOLO (You Only Look Once)** to detect objects in each video frame.  
- The model predicts bounding boxes and class scores, identifying objects present in the frame.  
- Large bounding boxes that take up most of the frame are ignored to reduce false detections.

#### 2. Appearance Feature Extraction
- For each detected object, we extract **appearance features** using **ROI Align** from an intermediate YOLO feature map.  
- Global average pooling and L2 normalization convert each feature map into a compact descriptor.


#### 3. Object Tracking  
- We use a combination of **appearance features** and **motion prediction** to track objects across video frames.  

- **Steps involved:**

  1. **Cosine Similarity Calculation**:  
     - We compare the appearance descriptors of current detections and previous tracks using **cosine similarity**.  
     - This helps identify whether two objects look the same across frames.  
  2. **Motion Prediction via Kalman Filter**:  
     - Each track maintains a **Kalman filter** to predict its next position.  
     - This provides robustness to missed detections or brief occlusions.  
  3. **Spatial Similarity (IoU)**:  
     - We compute **IoU** between predicted track positions and current detections to measure spatial consistency.  
  4. **Cost Matrix + Hungarian Assignment**:  
     - A weighted combination of appearance and spatial similarity forms a **cost matrix**.  
     - The **Hungarian algorithm** assigns detections to tracks by minimizing cost.  
  5. **Track Update and Management**:  
     - Assigned tracks are updated with new detections and refined Kalman estimates.  
     - Lost tracks are kept alive for a limited number of frames (e.g. 30), then removed.  
     - New unmatched detections are given new IDs and initialized as new tracks.


## Requirements

### 1. Create and Activate Conda Environment
```sh
conda create -n object_tracking python=3.9 -y
conda activate object_tracking
```

### 2. Install Dependencies
```sh
conda install -c conda-forge opencv numpy scipy
pip install torch torchvision torchaudio ultralytics
```

## Running the Script
To run the object tracking system on a video:
```sh
python main.py
```



That's it! You can now run the object tracking system on your own videos. The script will process the video, detect objects, extract features, and track them across frames. The output will be saved as a new video file with bounding boxes and IDs drawn around the tracked objects.

Contributions and improvements are welcome! Feel free to modify the code, add new features, or enhance the tracking algorithm. If you have any questions or suggestions, please open an issue or submit a pull request.

Happy tracking! 
