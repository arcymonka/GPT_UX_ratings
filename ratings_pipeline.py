import os
from dotenv import load_dotenv
import openai
import cv2
# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

video_path = os.getenv("VIDEO_PATH")  

# Define the prompt to process the text
prompt = f"""
You are a passenger in an autonomous car. You’re  years old. 
Watch the following video that shows what happens in the surroundings and the driving situation and rate your reaction to what you see 
based on the following questions. Use the scales provided in the brackets for each question. When there are two adjectives to compare the 
left value corresponds to the left adjective and the right value to the right one. Don’t respond with an analysis or any other comments 
(give just the ratings). Provide the response in a csv file.
"""

def extract_frames_quarter_second(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    name = os.path.splitext(os.path.basename(video_path))[0]
    name_folder = os.path.join(output_folder, name)

    if not os.path.exists(name_folder):
        os.makedirs(name_folder)

    for quarter_second in range(int(duration * 4)):
        target_frame = quarter_second * (fps / 4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(target_frame))
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(name_folder, f"{name}_frame_{quarter_second}.jpg")
            cv2.imwrite(frame_filename, frame)

    cap.release()


# Function to process image frame with OpenAI (e.g., using CLIP for image/text analysis)
def process_frame_with_openai(image_path):
    try:
        with open(image_path, "rb") as image_file:
            response = openai.Image.create(
                prompt=prompt,
                model="pick the model",
                n=1,
                size="1024x1024",
                file=image_file
            )

            # Extract and return the description from OpenAI's response
            return response['data'][0]['text']

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process the video: Extract frames and send them to OpenAI
def process_video_with_openai(video_path, output_folder):
    # Step 1: Extract frames from the video
    extract_frames_quarter_second(video_path, output_folder)

    # Step 2: Process each frame with OpenAI
    for frame_filename in os.listdir(output_folder):
        frame_path = os.path.join(output_folder, frame_filename)

        
        # Process the frame with OpenAI (e.g., CLIP for image/text analysis)
        description = process_frame_with_openai(frame_path)

        if description:
            print(f"Description for {frame_filename}: {description}")
        else:
            print(f"Failed to process {frame_filename}")



# ratings_folder = 'ratings'

# video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mkv', '.avi'))]

# video_files = [os.path.join(folder_path, video) for video in video_files]

# age_list = ['18-24', '25-34', '35-44', '45-54', '55-64']

# for file in video_files:
#     for age in age_list:
#         age = age
#     print(f"Processing video: {file}")
#     process_video_with_openai(file, ratings_folder)
#     print(f"Finished processing video: {file}")



