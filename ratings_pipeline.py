import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Prompt Template ===
def build_prompt(summary, age, gender):
    return f"""
You are a passenger in an autonomous car. You’re {age} years old and your gender is {gender}. 
Based on the following summary of a driving situation, rate your reaction using the scales provided. 
**ONLY** use the numerical ratings – do not add any text or explanations. Return the result as a CSV row (no header, just the numbers, separated by commas).

Summary:
\"\"\"
{summary}
\"\"\"

The questions are:
• How mentally demanding was the task? (1 to 20)
• anxious – relaxed (−3 to +3)
• agitated – calm (−3 to +3)
• unsafe – safe (−3 to +3)
• timid – confident (−3 to +3)
• I trust the highly automated vehicle. (1 to 5)
• I can rely on the highly automated vehicle. (1 to 5)
• The system state was always clear to me. (1 to 5)
• The system reacts unpredictably. (1 to 5)
• I was able to understand why things happened. (1 to 5)
• It’s difficult to identify what the system will do next. (1 to 5)
• useful – useless (1 to 7)
• pleasant – unpleasant (1 to 7)
• bad – good (1 to 7)
• nice – annoying (1 to 7)
• effective – superfluous (1 to 7)
• irritating – likeable (1 to 7)
• assisting – worthless (1 to 7)
• undesirable – desirable (1 to 7)
• raising alertness – sleep-inducing (1 to 7)
• How changeable is the situation? (1 to 7)
• How complicated is the situation? (1 to 7)
• How many variables are changing? (1 to 7)
• How aroused are you? (1 to 7)
• How much are you concentrating? (1 to 7)
• How much is your attention divided? (1 to 7)
• How much mental capacity do you have to spare? (1 to 7)
• How much information have you gained? (1 to 7)
• How good is the information you have gained? (1 to 7)
• How familiar are you with the situation? (1 to 7)
"""

import re

def sanitize_csv_row(raw_text: str, expected_cols: int = 30) -> str | None:
    """
    Extract strictly numeric values from model output and enforce a fixed column count.
    - Converts Unicode minus (−) to ASCII '-'
    - Accepts integers or decimals with optional leading sign
    - Truncates if there are too many; pads with empty fields if too few
    Returns a comma-separated string or None if nothing usable.
    """
    if not raw_text:
        return None

    # Normalize unicode minus and strip spaces/newlines
    text = raw_text.replace("−", "-").strip()

    # Pull out numbers like -3, +2, 4.5, -0.25
    nums = re.findall(r"[+\-]?\d+(?:\.\d+)?", text)

    if not nums:
        return None

    # Enforce column count
    if len(nums) > expected_cols:
        nums = nums[:expected_cols]
    elif len(nums) < expected_cols:
        nums = nums + [""] * (expected_cols - len(nums))

    # Return CSV line without spaces
    return ",".join(nums)


# === OpenAI Call ===
def process_summary_with_openai(summary_text, age, gender):
    prompt = build_prompt(summary_text, age, gender)
    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano",  # pick a real model from the docs
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        return sanitize_csv_row(raw, expected_cols=30)
    except Exception as e:
        print(f"Error processing summary: {e}")
        return None



# === Main Summary Folder Processor ===
def process_all_summaries(input_folder, output_folder):
    age_list = [25]
    gender_list = ["male", "female"]

    summary_files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".txt") or f.endswith(".md")
    ]

    for summary_file in summary_files:
        summary_path = os.path.join(input_folder, summary_file)
        with open(summary_path, "r") as f:
            summary_text = f.read()

        summary_name = os.path.splitext(summary_file)[0]

        for age in age_list:
            for gender in gender_list:
                print(f"🟢 Processing {summary_file} for age {age}, gender {gender}")
                csv_response = process_summary_with_openai(summary_text, age, gender)
                if csv_response:
                    csv_filename = f"{summary_name}_age_{age}_{gender}.csv"
                    csv_path = os.path.join(output_folder, csv_filename)
                    with open(csv_path, "w") as out_f:
                        out_f.write(csv_response + "\n")
                    print(f"✅ Saved: {csv_filename}")
                else:
                    print(f"❌ Failed to process {summary_file} for age {age}, gender {gender}")

# === Run Script ===
if __name__ == "__main__":
    input_folder = os.getenv("SUMMARY_PATH")
    output_folder = os.getenv("RATINGS_OUTPUT_PATH")
    os.makedirs(output_folder, exist_ok=True)

    process_all_summaries(input_folder, output_folder)
