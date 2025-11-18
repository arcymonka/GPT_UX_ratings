# GPT_UX_Ratings

<p align="center"> A pipeline for generating synthetic UX ratings for autonomous-vehicle traffic scenarios using GPT-5. </p> <p align="center"> <img src="https://img.shields.io/badge/openai-0.27.0-blue" /> <img src="https://img.shields.io/badge/pandas-2.1.3-yellow" /> <img src="https://img.shields.io/badge/opencv--python-4.5.5.64-red" /> <img src="https://img.shields.io/badge/python--dotenv-0.21.1-green" /> </p>


## Table of Contents
1. [Introduction](#1-introduction)
2. [Features](#2--features)
3. [Project Structure](#3-project-structure)
4. [Setup Instructions](#4-setup-instructions)
   1. [Clone the Repository](#1-clone-the-repository)
   2. [Configuration](#2-configuration)
5. [Usage](#5-usage)
6. [Questionnaire Sources](#6-questionnaire-sources)
7. [Troubleshooting](#7-troubleshooting)
8. [License](#8-license)
9. [References](#9-references)



## 1. Introduction

**GPT UX Ratings** is designed to evaluate how different traffic situations may be perceived by passengers in autonomous vehicles. The system uses OpenAI's GPT model to simulate answers to established UX rating questions.

This method helps generate synthetic datasets for traffic safety, automation research, and behavioral studies involving passenger perceptions.

## 2.  Features

This project uses **OpenAI’s GPT-5** model to analyze frames extracted from driving videos and simulate human emotional reactions (as CSV ratings) for different age groups and gender.

It automates the following:
1. Extracts frames every **¼ second** from each video.
2. Sends the frames in batches of 10 in  **GPT-5** API.
3. Prompts the model to generate a summary of the videos events  
4. Prompts the model to generate a participant rating of the situations based on the summaries 
5. Saves all responses as **CSV files**.

  
## 3. Project Structure

```plaintext
.
├── ratings_pipeline.py     # Main processing script
├── .env                    # Environment variables
├── requirements.txt         # Python dependencies (recommended)
└── output_extract/          # Folder where extracted frames and CSVs are saved
```

## 4. Setup Instructions

### 1. Clone the Repository
Clone the repository and install dependencies:

```bash
git clone https://github.com/arcymonka/GPT_UX_ratings.git
cd GPT_UX_ratings
pip install -r requirements.txt
```
### 2. Configuration

Create a `.env` file in the root directory (or use the provided one) to configure environment variables:

```ini
OPENAI_API_KEY="your_openai_key"
VIDEO_PATH="path/to/videos"
OUTPUT_PATH="path/to/frames"
SUMMARY_PATH="path/to/summaries"
RATINGS_OUTPUT_PATH="path/to/output/ratings"
RANDOM_SEED=42


Ensure all referenced directories exist and contain valid data files (e.g., `.txt` summaries in `SUMMARY_PATH`).
```

## 5. Usage

Run the scripts in the following order: 
```bash 
python frames.py
python ratings_pipeline.py
python part_rat.py
```


By default, the scripts will:

- Load image frames from the `frames/` directory.  
- Process each frame using the OpenAI API.  
- Read each summary file from the configured `SUMMARY_PATH`
- Generate a prompt for each age/gender combination
- Call OpenAI's API to simulate ratings
- Save each response to a CSV in `RATINGS_OUTPUT_PATH`


## 6. Questionnaire Sources

The 30-item rating scale is built upon validated measures from several academic sources:

- **Perceived Safety** – Faas et al. (2020)
- **Trust in Automation** – Körber (2019)
- **Predictability** – Körber (2019)
- **Acceptance** – Van der Laan et al. (1997)
- **SART (Situation Awareness Rating Technique)** – Taylor (2017)

## 7. Troubleshooting

| Issue                             | Possible Cause                                | Solution                                   |
|----------------------------------|-----------------------------------------------|--------------------------------------------|
| No CSV output                    | API key issue or missing input files          | Check `.env` values and summary folder     |
| Ratings are misformatted         | Unexpected model output                       | Review prompt and ensure GPT model validity|
| API error or rate limit exceeded | Too many requests or invalid model version    | Try again later or adjust request volume or add money to API account |
| Output folder not created        | Missing permissions or invalid path           | Ensure script can create/write to paths    |

## 8.. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 9. References 

Anthis, J. R., Liu, R., Richardson, S. M., Kozlowski, A. C., Koch, B., Brynjolfsson, E., Evans, J., & Bernstein, M. S. (2025). LLM social simulations are a promising research method (arXiv preprint arXiv:2504.02234 v2). arXiv. https://doi.org/10.48550/arXiv.2504.02234

Faas, S. M., Mattes, S., Kao, A. C., & Baumann, M. (2020). Efficient paradigm to measure street-crossing onset time of pedestrians in video-based interactions with vehicles. Information, 11(7), 360. https://doi.org/10.3390/info11070360

Körber, M. (2019). Theoretical considerations and development of a questionnaire to measure trust in automation. In S. Bagnara, R. Tartaglia, S. Albolino, T. Alexander, & Y. Fujita (Eds.), Proceedings of the 20th Congress of the International Ergonomics Association (IEA 2018) (Advances in Intelligent Systems and Computing, Vol. 823, pp. 13-30). Springer. https://doi.org/10.1007/978-3-319-96074-6_2

Taylor, R. M. (2017). Situational Awareness Rating Technique (SART): The Development of a Tool for Aircrew Systems Design. In Situational Awareness (pp. 111–128). Routledge. https://doi.org/10.4324/9781315087924-8

Van Der Laan, J. D., Heino, A., & De Waard, D. (1997). A simple procedure for the assessment of acceptance of advanced transport telematics. Transportation Research Part C: Emerging Technologies, 5(1), 1–10. https://doi.org/10.1016/S0968-090X(96)00025-3
