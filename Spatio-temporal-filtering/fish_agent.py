import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
from glob import glob

class FishBehaviorAgent:
    def __init__(self, video_path, frame_skip=1, frames_dir="frames", flow_vis_dir="flow_vis"):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.frames_dir = frames_dir
        self.flow_vis_dir = flow_vis_dir

    def extract_frames(self):
        """Extract frames from the video, save them as images to disk."""
        if os.path.exists(self.frames_dir):
            existing_files = [f for f in os.listdir(self.frames_dir) if f.endswith(".png")]
            if existing_files:
                print(f"Frames already exist in '{self.frames_dir}'. Skipping extraction.")
                return

        os.makedirs(self.frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {self.video_path}")

        frame_idx, saved_count = 0, 0
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
        """Compute dense optical flow between saved frames and visualize/save results."""
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

            # HSV visualization
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

    def analyze_flows(self):
        """Analyze motion patterns: average speed, directions, heatmap, and export results."""
        print("Analyzing optical flows...")

        frame_files = sorted(glob(os.path.join(self.frames_dir, "*.png")))
        if len(frame_files) < 2:
            raise ValueError("Not enough frames for analysis.")

        avg_speeds, all_angles = [], []
        heatmap = None

        for i in range(1, len(frame_files)):
            frame = cv2.imread(frame_files[i])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(cv2.imread(frame_files[i - 1]), cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_speed = np.mean(magnitude)
            avg_speeds.append(avg_speed)

            # Memory-safe: sample up to 1000 angles per frame
            angle_flat = angle.flatten()
            sample_size = min(1000, len(angle_flat))
            sample_indices = np.random.choice(len(angle_flat), size=sample_size, replace=False)
            sampled_angles = angle_flat[sample_indices]
            all_angles.extend(sampled_angles)

            if heatmap is None:
                heatmap = magnitude.copy()
            else:
                heatmap += magnitude

            if i > 1:
                speed_diff = abs(avg_speeds[-1] - avg_speeds[-2])
                if speed_diff > 5:  # tweak threshold if needed
                    print(f">> Sudden change detected at frame {i}! Possible snapping/grazing.")

        # Plot average swim speed over time
        plt.figure(figsize=(10, 4))
        plt.plot(avg_speeds, label="Average swim speed")
        plt.xlabel("Frame")
        plt.ylabel("Average speed (pixels/frame)")
        plt.title("Fish Average Swim Speed Over Time")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot direction histogram
        angles_deg = (np.degrees(np.array(all_angles)) + 360) % 360
        plt.figure(figsize=(8, 4))
        plt.hist(angles_deg, bins=36, range=(0, 360), color='purple', alpha=0.7)
        plt.xlabel("Direction (degrees)")
        plt.ylabel("Frequency")
        plt.title("Preferred Swim Directions Histogram")
        plt.grid()
        plt.show()

        # Plot heatmap of movement
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imshow("Movement Heatmap", colored_heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Analysis complete!")

        # Export results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # 1) Export average speed time series to CSV
        df_speed = pd.DataFrame({
            "frame_idx": list(range(1, len(avg_speeds) + 1)),
            "average_speed": avg_speeds
        })
        csv_speed_path = os.path.join(results_dir, "average_speeds.csv")
        df_speed.to_csv(csv_speed_path, index=False)
        print(f"Average swim speeds saved to {csv_speed_path}")

        # 2) Export direction histogram to JSON
        hist, bin_edges = np.histogram(angles_deg, bins=36, range=(0, 360))
        hist_data = {
            "angle_bins": bin_edges.tolist(),
            "counts": hist.tolist()
        }
        json_hist_path = os.path.join(results_dir, "direction_histogram.json")
        with open(json_hist_path, "w") as f:
            json.dump(hist_data, f, indent=2)
        print(f"Direction histogram saved to {json_hist_path}")

        # 3) Save analysis metadata to JSON
        metadata = {
            "video_path": self.video_path,
            "frame_skip": self.frame_skip,
            "frames_analyzed": len(avg_speeds),
        }
        json_meta_path = os.path.join(results_dir, "analysis_metadata.json")
        with open(json_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Analysis metadata saved to {json_meta_path}")

    def run(self):
        """Run full pipeline: frame extraction, optical flow, analysis."""
        print("Checking frame folder...")
        self.extract_frames()
        print("Frame extraction done (or skipped).")

        print("Starting optical flow computation from saved frames...")
        self.compute_optical_flow(visualize=False, save_vis=True)
        print("Optical flow computation done.")

        print("Starting analysis of optical flow...")
        self.analyze_flows()

if __name__ == "__main__":
    agent = FishBehaviorAgent(
        r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\Numpy-Keras-Seaborn-and-LSTM-1\datasets\test_video.mp4",
        frame_skip=2
    )
    agent.run()
