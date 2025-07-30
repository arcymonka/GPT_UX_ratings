import os
import base64
import cv2
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt template
def build_prompt(age):
    return f"""
    You are a passenger in an autonomous car. You’re {age} years old. 
    Watch the following images (from a video that shows what happens in the surroundings and the driving situation) 
    and rate your reaction to what you see based on the following questions. 
    Use the scales provided in the brackets for each question. 
    When there are two adjectives to compare, the left value corresponds to the left adjective 
    and the right value to the right one. Don’t respond with an analysis or any other comments 
    (give just the ratings). Provide the response in a CSV file format.
    """


# === Frame Extraction ===
def extract_frames_quarter_second(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    name = os.path.splitext(os.path.basename(video_path))[0]
    name_folder = os.path.join(output_folder, name)
    os.makedirs(name_folder, exist_ok=True)

    for quarter_second in range(int(duration * 4)):
        target_frame = quarter_second * (fps / 4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(target_frame))
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(name_folder, f"{name}_frame_{quarter_second}.jpg")
            cv2.imwrite(frame_filename, frame)

    cap.release()

# === Frame Encoding ===
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === GPT-4 Vision Batch Processing ===
def process_frames_with_openai(frames, age):
    # Limit to 20 images (API limitation)
    frames = frames[:20]

    images_payload = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(frame)}"
            }
        }
        for frame in frames
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt(age)},
                        *images_payload
                    ]
                }
            ],
            max_tokens=1000
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"Error processing frames: {e}")
        return None

# === Main Video Folder Processor ===
def process_all_videos(video_dir, output_folder):
    age_list = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith((".mp4", ".mkv", ".avi"))
    ]

    for video_file in video_files:
        print(f"🟡 Extracting frames from {video_file}")
        extract_frames_quarter_second(video_file, output_folder)

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        frame_folder = os.path.join(output_folder, video_name)

        # Collect all frame paths
        frame_paths = [
            os.path.join(frame_folder, f)
            for f in sorted(os.listdir(frame_folder))
            if f.endswith(".jpg")
        ]

        if not frame_paths:
            print(f"⚠️ No frames found in {frame_folder}")
            continue

        print(f"🟢 Sending frames for {video_name} to OpenAI...")
        for age in age_list:
            description = process_frames_with_openai(frame_paths, age)

            if description:
                print(f"✅ CSV Ratings for {video_name}:\n{description}\n")
                csv_filename = f"{video_name}_age_{age}_ratings.csv"
                csv_path = os.path.join(output_folder, csv_filename)
                with open(csv_path, "w") as f:
                    f.write(description)
            else:
                print(f"❌ Failed to process {video_name}\n")

# === Run Script ===
if __name__ == "__main__":
    video_dir = os.getenv("VIDEO_PATH")  # This must be a folder path
    output_folder = "/Users/helena/Desktop/vid/output_extract"
    process_all_videos(video_dir, output_folder)
