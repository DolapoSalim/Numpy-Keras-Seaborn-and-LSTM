import cv2
import os

class FishBehaviorAgent:
    def __init__(self, video_path, frame_skip=1):
        self.video_path = video_path
        self.frame_skip = frame_skip  # Process every nth frame to speed up if needed
        self.frames = []

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {self.video_path}")

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.frame_skip == 0:
                self.frames.append(frame)
                extracted_count += 1
            frame_count += 1

        cap.release()
        print(f"Extracted {extracted_count} frames from {frame_count} total frames.")

    def run(self):
        print("Starting frame extraction...")
        self.extract_frames()
        print("Frame extraction done.")

# Example usage
if __name__ == "__main__":
    agent = FishBehaviorAgent("your_fish_video.mp4", frame_skip=2)  # skip every other frame
    agent.run()
# Make sure to replace "your_fish_video.mp4" with the actual path to your video file.