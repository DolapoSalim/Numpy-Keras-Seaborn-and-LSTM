import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "your_fish_video.mp4"  # change to your actual file
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
if not ret:
    print("Cannot read video.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Storage for analysis
speeds = []
heatmap = np.zeros_like(prev_gray, dtype=np.float32)  # for movement density
all_angles = []  # for direction histogram

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Visualization: HSV flow image
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    dense_flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Dense Optical Flow", dense_flow_bgr)

    # === Analysis: average speed ===
    avg_speed = np.mean(magnitude)
    speeds.append(avg_speed)

    # Update heatmap: accumulate movement magnitude
    heatmap += magnitude

    # Store angles for histogram (flattened)
    all_angles.extend(angle.flatten())

    print(f"Frame {frame_count}: Average swim speed = {avg_speed:.2f} pixels/frame")

    # Detect sudden movement based on speed change
    if len(speeds) > 2:
        speed_diff = abs(speeds[-1] - speeds[-2])
        if speed_diff > 5:  # tune this threshold for your data
            print(f">> Sudden change detected at frame {frame_count}! Possible snapping/grazing.")

    prev_gray = gray.copy()
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === After processing video ===

# 1) Plot average swim speed over time
plt.figure(figsize=(10, 4))
plt.plot(speeds, label="Average swim speed")
plt.xlabel("Frame")
plt.ylabel("Average speed (pixels/frame)")
plt.title("Fish Average Swim Speed Over Time")
plt.legend()
plt.grid()
plt.show()

# 2) Show movement heatmap
heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = heatmap_norm.astype(np.uint8)
colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
cv2.imshow("Movement Heatmap", colored_heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3) Direction histogram
angles_deg = np.degrees(np.array(all_angles)).flatten()  # radians to degrees
angles_deg = (angles_deg + 360) % 360  # ensure positive angles 0-360

plt.figure(figsize=(8, 4))
plt.hist(angles_deg, bins=36, range=(0, 360), color='purple', alpha=0.7)
plt.xlabel("Direction (degrees)")
plt.ylabel("Frequency")
plt.title("Preferred Swim Directions Histogram")
plt.grid()
plt.show()
