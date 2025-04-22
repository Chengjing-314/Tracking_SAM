import pyrealsense2 as rs
import cv2
import time
import os
import numpy as np

# Parameters
duration_seconds = 10         # Total recording duration
fps = 30                     # Frames per second
output_dir = "captured_images"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, fps)

# Start streaming (but we'll wait before capturing)
pipeline.start(config)

# Wait for warm-up
print("Camera warming up...")
time.sleep(1)

# Show preview and wait for key press
print("Press any key to start recording...")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow("Preview - Press any key to start", color_image)

    key = cv2.waitKey(1)
    if key != -1:  # Any key pressed
        break

cv2.destroyAllWindows()
print(f"Recording for {duration_seconds} second(s) at {fps} FPS...")

# Start recording
start_time = time.time()
frame_count = 0

try:
    while True:
        current_time = time.time()
        if current_time - start_time > duration_seconds:
            break

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        filename = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(filename, color_image)
        print(f"Saved {filename}")
        frame_count += 1

        time.sleep(1 / fps)

finally:
    pipeline.stop()
    print("Done recording.")
