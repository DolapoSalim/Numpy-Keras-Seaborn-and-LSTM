import numpy as np
import cv2
import os
from glob import sorted, glob

class FishBehaviorAgent:
    def __init__(self, video_path, frame_skip=1, frames_dir="frames", flow_vis_dir="flow_vis"):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.frames_dir = frames_dir
        self.flow_vis_dir = flow_vis_dir

    def extract_frames(self):
        # Check if frame folder exists and has frames
        if os.path.exists(self.frames_dir):
            existing_files = [f for f in os.listdir(self.frames_dir) if f.endswith(".png")]
            if existing_files:
                print(f"Frames already exist in '{self.frames_dir}'. Skipping extraction.")
                return

        os.makedirs(self.frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {self.video_path}")

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                frame_filename = os.path.join(self.frames_dir, f"frame_{saved_count:05d}.png")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            frame_idx += 1

        cap.release()
        print(f"Extracted and saved {saved_count} frames in '{self.frames_dir}'.")

    def compute_optical_flow(self, visualize=True, save_vis=True):
        # Read all frames from the frames_dir
        frame_files = sorted(glob(os.path.join(self.frames_dir, "*.png")))
        if len(frame_files) < 2:
            raise ValueError("Need at least two frames to compute optical flow.")

        if save_vis:
            os.makedirs(self.flow_vis_dir, exist_ok=True)

        print("Computing dense optical flow from saved frames...")

        prev_frame = cv2.imread(frame_files[0])
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frame_files)):
            frame = cv2.imread(frame_files[i])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Create HSV visualization of flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            if visualize:
                cv2.imshow("Dense Optical Flow", flow_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_vis:
                flow_filename = os.path.join(self.flow_vis_dir, f"flow_{i:05d}.png")
                cv2.imwrite(flow_filename, flow_img)

            prev_gray = gray

        cv2.destroyAllWindows()
        print(f"Computed optical flow for {len(frame_files) - 1} frame pairs. Visualizations saved to '{self.flow_vis_dir}'.")

    def run(self):
        print("Checking frame folder...")
        self.extract_frames()
        print("Frame extraction done (or skipped).")

        print("Starting optical flow computation from saved frames...")
        self.compute_optical_flow(visualize=True, save_vis=True)
        print("Optical flow computation done.")