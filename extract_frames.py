import cv2
import os

INPUT_DIR = "videos"
OUTPUT_DIR = "dataset"
IMG_SIZE = 224
FRAMES_PER_VIDEO = 12   # keep between 10–15

def extract_frames(video_path, save_dir, label, video_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(total_frames // FRAMES_PER_VIDEO, 1)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0 and saved < FRAMES_PER_VIDEO:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            filename = f"{video_name}_frame_{saved}.jpg"
            save_path = os.path.join(save_dir, label, filename)

            cv2.imwrite(save_path, frame)
            saved += 1

        count += 1

    cap.release()


for label in ["violence", "non_violence"]:
    input_folder = os.path.join(INPUT_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)

    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)

        video_name = os.path.splitext(video_file)[0]

        print(f"Processing: {video_file}")
        extract_frames(video_path, OUTPUT_DIR, label, video_name)

print("Done extracting frames.")