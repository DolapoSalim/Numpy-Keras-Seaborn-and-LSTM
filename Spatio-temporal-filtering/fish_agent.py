import cv2
import os

class FishBehaviorAgent:
    def __init__(self, video_path, frame_skip=1, frames_dir="frames"):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.frames_dir = frames_dir

    def extract_frames(self):
        # Check if frame folder exists and has frames
        if os.path.exists(self.frames_dir):
            existing_files = [f for f in os.listdir(self.frames_dir) if f.endswith(".png")]
            if existing_files:
                print(f"Frames already exist in '{self.frames_dir}'. Skipping extraction.")
                return  # Skip extraction

        # Create directory if missing
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

    def run(self):
        print("Checking frame folder...")
        self.extract_frames()
        print("Frame extraction done (or skipped).")


# Example usage
if __name__ == "__main__":
    agent = FishBehaviorAgent(r"C:\\Users\\dolap\\OneDrive\\Documents\\DOLAPO\\data-analysis\\Numpy-Keras-Seaborn-and-LSTM-1\\datasets\\test_video.mp4", frame_skip=2)
    agent.run()