import cv2
import numpy as np

# === 1) Load your video ===
video_path = "your_fish_video.mp4"  # change to your actual file
cap = cv2.VideoCapture(video_path)

# === 2) Read first frame ===
ret, prev_frame = cap.read()
if not ret:
    print("Cannot read video.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Storage for analysis
speeds = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === 3) Calculate dense optical flow (Farneback method) ===
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Compute magnitude (speed) and angle (direction)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # === 4) Create HSV image for visualization ===
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255  # full saturation

    # Direction → hue; Speed → value
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    dense_flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Dense Optical Flow", dense_flow_bgr)

    # === 5) Analysis: average speed ===
    avg_speed = np.mean(magnitude)
    speeds.append(avg_speed)

    print(f"Average swim speed (frame): {avg_speed:.2f} pixels/frame")

    # === 6) Detect sudden movements ===
    if len(speeds) > 2:
        speed_diff = abs(speeds[-1] - speeds[-2])
        if speed_diff > 5:  # threshold to tune based on your video
            print(">> Sudden change detected! Possible snapping/grazing behavior.")

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()