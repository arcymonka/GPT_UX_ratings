# GPT_UX_Ratings

![OpenAI](https://img.shields.io/badge/openai-0.27.0-blue)
![python-dotenv](https://img.shields.io/badge/python--dotenv-0.21.1-green)
![pandas](https://img.shields.io/badge/pandas-2.1.3-yellow)
![opencv-python](https://img.shields.io/badge/opencv--python-4.5.5.64-red)


## Table of Contents
1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Setup Instructions](#3-setup-instructions)
   1. [Clone the Repository](#1-clone-the-repository)
   2. [Create a Virtual Environment](#1-create-a-virtual-environment)
4. [Usage](#1-usage)


   
## 1.  Overview

This project uses **OpenAI’s GPT-4 Vision** model to analyze frames extracted from driving videos and simulate human emotional reactions (as CSV ratings) for different age groups.

It automates the following:
1. Extracts frames every **¼ second** from each video.
2. Sends up to **20 frames per video** to the **GPT-4 Vision** API.
3. Prompts the model to rate the passenger’s reaction as if they were different ages (18–80).
4. Saves all responses as **CSV files**.


## 2. Project Structure
.
├── ratings_pipeline.py # Main processing script
├── .env # Environment variables
├── requirements.txt # Python dependencies (recommended)
└── output_extract/ # Folder where extracted frames and CSVs are saved


## 3. Setup Instructions

### 1. Clone the Repository
Clone the repository and install dependencies:

```bash
git clone https://github.com/arcymonka/GPT_UX_ratings.git
cd GPT_UX_ratings
pip install -r requirements.txt
```
### 2. Create a Virtual Environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## 4. Usage
Run the main script: 
```python ratings_pipeline.py
```

By default, the script will:

- Load image frames from the `frames/` directory.  
- Process each frame using the OpenAI API.  
- Generate and store UX ratings or analysis results in a structured format (e.g., CSV).

## References 
