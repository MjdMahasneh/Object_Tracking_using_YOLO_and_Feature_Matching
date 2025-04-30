import torch
import torchvision
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import torchvision.ops as ops
from filterpy.kalman import KalmanFilter
import time

# ✅ Configurable parameters
VIDEO_PATH = "./sample/video4.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"
IMG_SIZE = 640
ROI_ALIGN_SIZE = (7, 7)  # Increased from 2x2 for better feature representation
TARGET_LAYER_INDEX = -2  # Using second to last layer for feature extraction
MATCH_THRESHOLD = 0.5  # Threshold for appearance matching
MOTION_WEIGHT = 0.5  # Weight for motion prediction vs appearance (0-1)
APPEARANCE_WEIGHT = 0.5  # Weight for appearance similarity (0-1)
MAX_LOST_FRAMES = 30  # Frames before track is deleted
PLAYBACK_DELAY = 1  # Faster playback for debugging
DETECTION_CONFIDENCE = 0.4  # YOLO detection confidence
VIS_ACTIVE_TRACKS = True  # To visualize active tracks ONLY

# Initialize colors dictionary at the start
color_map = {}

# ✅ Load YOLO model
yolo_model = YOLO("yolo11n.pt")

# results = yolo_model.track(VIDEO_PATH, save=True, conf=DETECTION_CONFIDENCE)

# ✅ Hook to capture intermediate feature maps
FEATURE_MAPS = None


def hook_fn(module, input, output):
    global FEATURE_MAPS
    FEATURE_MAPS = output


# Choose a target layer with rich semantic information
target_layer = yolo_model.model.model[TARGET_LAYER_INDEX]
target_layer.register_forward_hook(hook_fn)

# ✅ Object tracking storage
object_tracks = {}
next_object_id = 1
tracking_started = False


def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())


# ✅ Kalman Filter initialization
def create_kalman_filter():
    """Initialize a Kalman filter for tracking in 4D state space: [x, y, w, h, vx, vy, vw, vh]"""
    kf = KalmanFilter(dim_x=8, dim_z=4)

    # State transition matrix (constant velocity model)
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
        [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
        [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
        [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
        [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
        [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
        [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
        [0, 0, 0, 0, 0, 0, 0, 1]  # vh = vh
    ])

    # Measurement matrix (we only measure x, y, w, h)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ])

    # Measurement noise
    kf.R = np.eye(4) * 10

    # Process noise
    kf.Q = np.eye(8) * 0.1
    kf.Q[4:, 4:] *= 0.01  # lower process noise for velocity components

    # Initial state covariance
    kf.P = np.eye(8) * 1000

    return kf


# Helper function to convert bbox to Kalman format and back
def bbox_to_z(bbox):
    """Convert bounding box to Kalman filter measurement: [x, y, w, h]"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return np.array([x, y, w, h])


def x_to_bbox(x):
    """Convert Kalman filter state to bounding box: [x1, y1, x2, y2]"""
    center_x, center_y, w, h = x[:4]
    x1 = int(center_x - w / 2)
    y1 = int(center_y - h / 2)
    x2 = int(center_x + w / 2)
    y2 = int(center_y + h / 2)
    return [x1, y1, x2, y2]


# Calculate IoU between boxes
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


# ✅ Process frame and extract features
def process_frame(frame):
    global FEATURE_MAPS
    orig_h, orig_w = frame.shape[:2]

    # Process with YOLO
    with torch.no_grad():
        results = yolo_model(frame, conf=DETECTION_CONFIDENCE)

    # Extract detections
    detections = results[0].boxes

    # Skip if no detections
    if len(detections) == 0:
        return [], [], frame

    # Convert bounding boxes to integer coordinates
    bbox_list = []
    confidence_scores = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Filter out too large objects (likely false positives)
        if (x2 - x1) < orig_w * 0.8 and (y2 - y1) < orig_h * 0.8:
            bbox_list.append([x1, y1, x2, y2])
            confidence_scores.append(conf)

    if len(bbox_list) == 0:
        return [], [], frame

    # Get feature map dimensions
    feature_maps = FEATURE_MAPS.cpu()
    C, H, W = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]

    # Calculate scaling factors between original image and feature map
    scale_x = W / orig_w
    scale_y = H / orig_h

    # Create correctly scaled boxes for ROI Align
    rois = []
    for box in bbox_list:
        x1, y1, x2, y2 = box
        # Scale coordinates to feature map size
        roi_x1 = max(0, x1 * scale_x)
        roi_y1 = max(0, y1 * scale_y)
        roi_x2 = min(W, x2 * scale_x)
        roi_y2 = min(H, y2 * scale_y)
        # Add batch index (0) to each ROI
        rois.append([0, roi_x1, roi_y1, roi_x2, roi_y2])

    # Convert to tensor for ROI Align
    try:
        if len(rois) == 0:
            return [], [], frame

        rois_tensor = torch.tensor(rois, dtype=torch.float32)

        # Extract features using ROI Align
        pooled_features = ops.roi_align(
            feature_maps,
            rois_tensor[:, 1:],  # Remove batch index
            output_size=ROI_ALIGN_SIZE,
            spatial_scale=1.0,  # Already scaled
            sampling_ratio=-1
        )

        # Global average pooling to get a feature vector per detection
        pooled_features = torch.mean(pooled_features, dim=[2, 3])

        # Normalize feature vectors for cosine similarity
        norms = torch.norm(pooled_features, p=2, dim=1, keepdim=True)
        pooled_features = pooled_features / (norms + 1e-5)

        return bbox_list, pooled_features.numpy(), frame

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        # Return empty features as fallback
        return bbox_list, np.zeros((len(bbox_list), C)), frame


# ✅ Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 20 == 0:
            elapsed = time.time() - start_time
            print(f"Processing frame {frame_count}, FPS: {frame_count / elapsed:.2f}")

        # Process the current frame
        bbox_list, features, frame = process_frame(frame)

        if len(bbox_list) == 0:
            out.write(frame)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord('q'):
                break
            continue

        if not tracking_started and len(bbox_list) > 0:
            tracking_started = True
            # Initialize first detections with Kalman filters
            for i, (bbox, feat) in enumerate(zip(bbox_list, features)):
                # Create and initialize Kalman filter
                kf = create_kalman_filter()
                # kf.x[:4] = bbox_to_z(bbox)  # Initialize state with current box
                kf.x[:4] = bbox_to_z(bbox).reshape(4, 1)  # Reshape to column vector

                object_tracks[next_object_id] = {
                    "bbox": bbox,
                    "features": feat,
                    "lost_frames": 0,
                    "color": random_color(),
                    "kf": kf,
                    "history": [bbox]  # Track history for visualization
                }
                color_map[next_object_id] = random_color()
                next_object_id += 1

        elif tracking_started:
            # First update all Kalman filters (prediction step)
            for track_id in object_tracks:
                object_tracks[track_id]["kf"].predict()

                # Get predicted bbox from Kalman state
                predicted_bbox = x_to_bbox(object_tracks[track_id]["kf"].x)
                object_tracks[track_id]["predicted_bbox"] = predicted_bbox

            # Build cost matrix combining appearance similarity and motion prediction
            cost_matrix = np.zeros((len(object_tracks), len(features)))
            object_ids = list(object_tracks.keys())

            # Calculate costs: combination of appearance similarity and spatial overlap
            for i, obj_id in enumerate(object_ids):
                predicted_bbox = object_tracks[obj_id]["predicted_bbox"]

                for j, (bbox, feat) in enumerate(zip(bbox_list, features)):
                    # Appearance similarity (using cosine similarity)
                    appearance_sim = 1 - cosine(object_tracks[obj_id]["features"], feat)

                    # Spatial similarity (using IoU between prediction and detection)
                    iou_sim = calculate_iou(predicted_bbox, bbox)

                    # Combined cost: weighted sum of appearance and motion
                    cost_matrix[i, j] = (APPEARANCE_WEIGHT * appearance_sim +
                                         MOTION_WEIGHT * iou_sim)

            # Consider only non-empty cases (i.e., if length of object_tracks and features are both > 0)
            if len(object_tracks) > 0 and len(features) > 0:
                # Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(-cost_matrix)

                # Filter assignments by threshold
                valid_assignments = []
                assigned_detections = set()

                for i, j in zip(row_ind, col_ind):
                    # Only consider matches above threshold
                    if cost_matrix[i, j] > MATCH_THRESHOLD:
                        obj_id = object_ids[i]
                        bbox = bbox_list[j]

                        # Update Kalman filter with new measurement
                        z = bbox_to_z(bbox)
                        object_tracks[obj_id]["kf"].update(z)

                        # Get corrected state estimate
                        corrected_bbox = x_to_bbox(object_tracks[obj_id]["kf"].x)

                        # Update the track
                        object_tracks[obj_id]["bbox"] = corrected_bbox

                        # Update appearance with smoothing
                        alpha = 0.8  # Weight for historical features
                        object_tracks[obj_id]["features"] = alpha * object_tracks[obj_id]["features"] + (1 - alpha) * \
                                                            features[j]

                        # Normalize features after update
                        object_tracks[obj_id]["features"] /= np.linalg.norm(object_tracks[obj_id]["features"]) + 1e-6

                        object_tracks[obj_id]["lost_frames"] = 0
                        object_tracks[obj_id]["history"].append(corrected_bbox)
                        if len(object_tracks[obj_id]["history"]) > 30:  # Limit history size
                            object_tracks[obj_id]["history"] = object_tracks[obj_id]["history"][-30:]

                        assigned_detections.add(j)
                        valid_assignments.append((obj_id, j))

                # Handle unmatched detections - create new tracks
                for j in range(len(features)):
                    if j not in assigned_detections:
                        # Create new Kalman filter
                        kf = create_kalman_filter()
                        # kf.x[:4] = bbox_to_z(bbox_list[j])
                        kf.x[:4] = bbox_to_z(bbox_list[j]).reshape(4, 1)

                        object_tracks[next_object_id] = {
                            "bbox": bbox_list[j],
                            "features": features[j],
                            "lost_frames": 0,
                            "color": random_color(),
                            "kf": kf,
                            "history": [bbox_list[j]]
                        }
                        color_map[next_object_id] = random_color()
                        next_object_id += 1

                # Increment lost counter for unmatched tracks
                for i, obj_id in enumerate(object_ids):
                    if not any(obj_id == matched_id for matched_id, _ in valid_assignments):
                        object_tracks[obj_id]["lost_frames"] += 1

                        # For lost tracks, we still update the bbox with Kalman prediction
                        # This helps maintain a more accurate track when detections are missed
                        object_tracks[obj_id]["bbox"] = object_tracks[obj_id]["predicted_bbox"]

                        # Add predicted position to history for visualization
                        object_tracks[obj_id]["history"].append(object_tracks[obj_id]["predicted_bbox"])
                        if len(object_tracks[obj_id]["history"]) > 30:
                            object_tracks[obj_id]["history"] = object_tracks[obj_id]["history"][-30:]

            else:
                # Handle edge cases when either no tracks or no detections
                if len(object_tracks) == 0 and len(features) > 0:
                    # All detections are new tracks
                    for j, (bbox, feat) in enumerate(zip(bbox_list, features)):
                        kf = create_kalman_filter()
                        kf.x[:4] = bbox_to_z(bbox)

                        object_tracks[next_object_id] = {
                            "bbox": bbox,
                            "features": feat,
                            "lost_frames": 0,
                            "color": random_color(),
                            "kf": kf,
                            "history": [bbox]
                        }
                        color_map[next_object_id] = random_color()
                        next_object_id += 1

                elif len(features) == 0 and len(object_tracks) > 0:
                    # All tracks are lost but still predicted
                    for obj_id in object_tracks:
                        object_tracks[obj_id]["lost_frames"] += 1
                        object_tracks[obj_id]["bbox"] = object_tracks[obj_id]["predicted_bbox"]
                        object_tracks[obj_id]["history"].append(object_tracks[obj_id]["predicted_bbox"])
                        if len(object_tracks[obj_id]["history"]) > 30:
                            object_tracks[obj_id]["history"] = object_tracks[obj_id]["history"][-30:]

        # Remove tracks that have been lost for too long
        to_delete = [obj_id for obj_id, track in object_tracks.items()
                     if track["lost_frames"] > MAX_LOST_FRAMES]
        for obj_id in to_delete:
            del object_tracks[obj_id]

        # Draw tracking results on the frame
        for obj_id, track in object_tracks.items():

            if VIS_ACTIVE_TRACKS:
                status = "Active" if track["lost_frames"] == 0 else f"Lost: {track['lost_frames']}"
                if status != "Active":
                    continue

            # Draw predicted bbox with dotted line (helpful for debugging)
            if "predicted_bbox" in track and track["lost_frames"] <= MAX_LOST_FRAMES:
                x1, y1, x2, y2 = track["predicted_bbox"]
                color = color_map.get(obj_id, random_color())
                # Draw dashed rectangle for predictions
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

            # Draw actual bbox with solid line
            if track["lost_frames"] <= MAX_LOST_FRAMES // 2:  # Show active tracks
                x1, y1, x2, y2 = track["bbox"]

                # Ensure color exists for this ID
                if obj_id not in color_map:
                    color_map[obj_id] = random_color()

                color = color_map[obj_id]

                # Draw bounding box (solid for active tracks)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add ID text and status
                status = "Active" if track["lost_frames"] == 0 else f"Lost: {track['lost_frames']}"
                cv2.putText(frame, f"ID:{obj_id} ({status})", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw trajectory (last N positions)
                if len(track["history"]) > 1:
                    # Extract centers of bounding boxes
                    centers = []
                    for h_bbox in track["history"]:
                        h_x1, h_y1, h_x2, h_y2 = h_bbox
                        center_x = int((h_x1 + h_x2) / 2)
                        center_y = int((h_y1 + h_y2) / 2)
                        centers.append((center_x, center_y))

                    # Draw polyline of trajectory
                    for i in range(1, len(centers)):
                        cv2.line(frame, centers[i - 1], centers[i], color, 2)

        # Visualize the track count
        active_tracks = sum(1 for track in object_tracks.values() if track["lost_frames"] == 0)
        cv2.putText(frame, f"Active Tracks: {active_tracks}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Tracks: {len(object_tracks)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Display the frame
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord('q'):
            break



except Exception as e:
    print(f"Error during processing: {e}")
    import traceback

    traceback.print_exc()
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Tracking complete. Output saved to {OUTPUT_VIDEO_PATH}")
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")