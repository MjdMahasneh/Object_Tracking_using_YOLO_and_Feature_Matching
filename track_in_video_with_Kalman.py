import torch
import torchvision
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import torchvision.ops as ops  # ROI pooling

# ✅ Configurable parameters
VIDEO_PATH = "./sample/video3.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"
IMG_SIZE = 640
ROI_ALIGN_SIZE = (2, 2)
TARGET_LAYER_INDEX = -2
MATCH_THRESHOLD = 0.7  # 🔄 Increased for better object matching
MAX_LOST_FRAMES = 50
PLAYBACK_DELAY = 100

# ✅ Load YOLO model
yolo_model = YOLO("yolo11n.pt")

# ✅ Hook to capture intermediate feature maps
FEATURE_MAPS = None

def hook_fn(module, input, output):
    global FEATURE_MAPS
    FEATURE_MAPS = output

target_layer = yolo_model.model.model[TARGET_LAYER_INDEX]
target_layer.register_forward_hook(hook_fn)

# ✅ Object tracking storage
object_tracks = {}
kalman_filters = {}
next_object_id = 1
tracking_started = False

# ✅ Kalman Filter Initialization (Tuned)
def init_kalman_filter(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  # 🔄 Reduced noise to avoid excessive drift
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1  # 🔄 Improved stability

    # Initialize with detected position
    kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
    return kf

# ✅ Process frame and extract features
def process_frame(frame):
    global FEATURE_MAPS
    orig_h, orig_w = frame.shape[:2]

    with torch.no_grad():
        results = yolo_model(frame)

    detections = results[0].boxes
    bbox_list = [list(map(int, box.xyxy[0])) for box in detections]

    if not bbox_list:
        return [], [], frame

    # ✅ Validate bounding box sizes
    bbox_list = [b for b in bbox_list if (b[2] - b[0]) < orig_w * 0.8 and (b[3] - b[1]) < orig_h * 0.8]

    if len(bbox_list) == 0:
        return [], [], frame

    yolo_h, yolo_w = results[0].orig_shape
    feature_maps = FEATURE_MAPS.cpu()
    fmap_h, fmap_w = feature_maps.shape[2], feature_maps.shape[3]

    scaled_bboxes = [
        [max(0, int(x1 / orig_w * fmap_w)), max(0, int(y1 / orig_h * fmap_h)),
         max(0, int(x2 / orig_w * fmap_w)), max(0, int(y2 / orig_h * fmap_h))]
        for x1, y1, x2, y2 in bbox_list
    ]

    if not scaled_bboxes:
        return bbox_list, np.zeros((len(bbox_list), 128)), frame

    scaled_bboxes = torch.tensor(scaled_bboxes, dtype=torch.float32)
    pooled_features = ops.roi_align(feature_maps, [scaled_bboxes], output_size=ROI_ALIGN_SIZE)
    pooled_features = pooled_features.view(pooled_features.shape[0], -1)

    return bbox_list, pooled_features.numpy(), frame

# ✅ Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    bbox_list, features, frame = process_frame(frame)

    if len(bbox_list) == 0:
        continue

    if not tracking_started:
        tracking_started = True

    cost_matrix = np.zeros((len(object_tracks), len(features)))

    object_ids = list(object_tracks.keys())
    col_ind = []

    if object_tracks and features.any():
        for i, obj_id in enumerate(object_ids):
            for j, feat in enumerate(features):
                cost_matrix[i, j] = 1 - cosine(object_tracks[obj_id]["features"], feat)

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        assignments = [
            (object_ids[i], bbox_list[j], features[j])
            for i, j in zip(row_ind, col_ind)
            if cost_matrix[i, j] > MATCH_THRESHOLD
        ]
    else:
        assignments = []

    assigned_objects = set()
    for obj_id, bbox, feature in assignments:
        x1, y1, x2, y2 = bbox
        object_tracks[obj_id] = {"bbox": bbox, "features": feature, "lost_frames": 0}
        assigned_objects.add(obj_id)

        if obj_id in kalman_filters:
            kf = kalman_filters[obj_id]
        else:
            kf = init_kalman_filter(x1, y1)
            kalman_filters[obj_id] = kf

        kf.correct(np.array([[np.float32(x1)], [np.float32(y1)]]))
        predicted = kf.predict()

        object_tracks[obj_id]["bbox"] = [int(predicted[0]), int(predicted[1]), x2, y2]  # ✅ Preserve size

    for obj_id in list(object_tracks.keys()):
        if obj_id not in assigned_objects:
            object_tracks[obj_id]["lost_frames"] += 1
        else:
            object_tracks[obj_id]["lost_frames"] = 0

    new_objects = set(range(len(bbox_list))) - set(col_ind)
    for i in new_objects:
        x1, y1, x2, y2 = bbox_list[i]
        object_tracks[next_object_id] = {"bbox": bbox_list[i], "features": features[i], "lost_frames": 0}
        kalman_filters[next_object_id] = init_kalman_filter(x1, y1)
        next_object_id += 1

    to_delete = [obj_id for obj_id, track in object_tracks.items() if track["lost_frames"] > MAX_LOST_FRAMES]
    for obj_id in to_delete:
        del object_tracks[obj_id]
        del kalman_filters[obj_id]

    def random_color():
        return tuple(np.random.randint(0, 255, 3).tolist())

    match_colors = {obj_id: random_color() for obj_id in object_tracks}

    for obj_id, track in object_tracks.items():
        x1, y1, x2, y2 = track["bbox"]
        color = match_colors[obj_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Tracking complete. Output saved to {OUTPUT_VIDEO_PATH}")
