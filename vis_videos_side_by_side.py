import cv2

# Paths to the videos
video1_path = "./runs/output.mp4"
video2_path = "./runs/video4.avi"
output_path = "./runs/side_by_side_output.mp4"

# Open video files
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both video files.")
    exit()

# Get properties from the first video
frame_width = 640 * 2  # Combined width of two videos
frame_height = 360  # Height of the resized frames
fps = int(cap1.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Read frames from both videos
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Break the loop if any video ends
    if not ret1 or not ret2:
        break

    # Resize frames to the same size (optional, for alignment)
    frame1 = cv2.resize(frame1, (640, 360))
    frame2 = cv2.resize(frame2, (640, 360))

    # Concatenate frames horizontally
    combined_frame = cv2.hconcat([frame1, frame2])

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Display the combined frame
    cv2.imshow("Side by Side Videos", combined_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and writer, and close windows
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()