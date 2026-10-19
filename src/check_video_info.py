
import cv2

video_path = "data/videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

if fps > 0:
    print("Video duration:", round(total_frames / fps, 2), "seconds")

cap.release()
