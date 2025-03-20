import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import torchvision.ops as ops  # ROI pooling

# ✅ Configurable parameters
IMG_SIZE = 640 #1280  # Set to 1280 or 640 (should work properly now)
ROI_ALIGN_SIZE = (2, 2)  # ROI Align output size
TARGET_LAYER_INDEX = -2  # Target feature extraction layer
MATCH_THRESHOLD = 0.5  # Cosine similarity threshold
IMAGE_PATHS = ["./sample/3.png", "./sample/4.png"]  # Modify paths here

# ✅ Load YOLO model (single model for detection & feature extraction)
yolo_model = YOLO("yolo11n.pt")

# ✅ Hook to capture intermediate feature maps
FEATURE_MAPS = None
def hook_fn(module, input, output):
    global FEATURE_MAPS
    FEATURE_MAPS = output

# Attach hook to the target feature layer
target_layer = yolo_model.model.model[TARGET_LAYER_INDEX]
target_layer.register_forward_hook(hook_fn)

# ✅ Function to process image and extract features
def process_image(image_path):
    global FEATURE_MAPS

    # Load image in original resolution
    img_orig = cv2.imread(image_path)
    orig_h, orig_w = img_orig.shape[:2]

    # ✅ Run YOLO inference on the **original image** (no pre-resizing)
    with torch.no_grad():
        results = yolo_model(image_path)

    # ✅ Get YOLO-detected bounding boxes in resized image coordinates
    detections = results[0].boxes
    bbox_list = [list(map(int, box.xyxy[0])) for box in detections]

    # ✅ Get actual shape YOLO processed the image at
    yolo_h, yolo_w = results[0].orig_shape  # **Fix: Get true YOLO processing shape**
    print(f"✅ YOLO Processed Image Shape: {yolo_h}x{yolo_w}")

    # ✅ Feature maps captured via the hook
    feature_maps = FEATURE_MAPS.cpu()  # Shape: (1, C, H, W)
    fmap_h, fmap_w = feature_maps.shape[2], feature_maps.shape[3]

    # ✅ Convert bounding boxes **from YOLO-sized image back to the original image**
    def rescale_bbox(bbox):
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * (orig_w / yolo_w))  # Use actual YOLO width
        x2 = int(x2 * (orig_w / yolo_w))
        y1 = int(y1 * (orig_h / yolo_h))  # Use actual YOLO height
        y2 = int(y2 * (orig_h / yolo_h))
        return [x1, y1, x2, y2]

    bbox_list = [rescale_bbox(b) for b in bbox_list]

    # ✅ Convert bounding boxes to feature map scale for ROI Align
    scaled_bboxes = []
    for x1, y1, x2, y2 in bbox_list:
        x1_f = max(0, int(x1 / orig_w * fmap_w))
        x2_f = max(0, int(x2 / orig_w * fmap_w))
        y1_f = max(0, int(y1 / orig_h * fmap_h))
        y2_f = max(0, int(y2 / orig_h * fmap_h))
        scaled_bboxes.append([x1_f, y1_f, x2_f, y2_f])

    if not scaled_bboxes:
        print("❌ No bounding boxes detected!")
        return [], [], img_orig

    # Convert bounding boxes to tensor for ROI Align
    scaled_bboxes = torch.tensor(scaled_bboxes, dtype=torch.float32)

    # ✅ Extract per-object features using ROI Align
    pooled_features = ops.roi_align(feature_maps, [scaled_bboxes], output_size=ROI_ALIGN_SIZE)
    pooled_features = pooled_features.view(pooled_features.shape[0], -1)  # Flatten to [N, C]

    return bbox_list, pooled_features.numpy(), img_orig

# ✅ Process both images
bbox_list1, features_img1, img1_orig = process_image(IMAGE_PATHS[0])
bbox_list2, features_img2, img2_orig = process_image(IMAGE_PATHS[1])

# ✅ Match objects using cosine similarity
matched_boxes = []
used_bboxes = set()

for feat1, bbox1 in zip(features_img1, bbox_list1):
    best_match = None
    best_similarity = -1

    for feat2, bbox2 in zip(features_img2, bbox_list2):
        similarity = 1 - cosine(feat1, feat2)

        # **Fix: Convert `bbox2` to a tuple before checking set**
        if similarity > best_similarity and tuple(bbox2) not in used_bboxes:
            best_similarity = similarity
            best_match = bbox2

    if best_match and best_similarity > MATCH_THRESHOLD:
        matched_boxes.append((bbox1, best_match, best_similarity))
        used_bboxes.add(tuple(best_match))

# ✅ Generate colors for each match
def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

match_colors = [random_color() for _ in matched_boxes]

# ✅ Draw bounding boxes for first image
for i, (bbox1, bbox2, similarity) in enumerate(matched_boxes):
    x1, y1, x2, y2 = bbox1
    color = match_colors[i]

    cv2.rectangle(img1_orig, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img1_orig, f"ID {i+1} ({round(similarity, 2)})", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ✅ Draw bounding boxes for second image
for i, (bbox1, bbox2, similarity) in enumerate(matched_boxes):
    x1_m, y1_m, x2_m, y2_m = bbox2
    color = match_colors[i]

    cv2.rectangle(img2_orig, (x1_m, y1_m), (x2_m, y2_m), color, 2)
    cv2.putText(img2_orig, f"ID {i+1} ({round(similarity, 2)})", (x1_m, y1_m - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ✅ Convert images from BGR (CV2 format) to RGB (Matplotlib format)
img1_rgb = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB)

# ✅ Show images side by side using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img1_rgb)
axes[0].set_title("Image 1 (Original Size)")
axes[0].axis("off")

axes[1].imshow(img2_rgb)
axes[1].set_title("Image 2 (Original Size)")
axes[1].axis("off")

plt.show()
