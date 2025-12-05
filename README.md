# GPT_UX_Ratings

<p align="center"> A pipeline for generating synthetic UX ratings for autonomous-vehicle traffic scenarios using GPT-5.1 </p> <p align="center"> <img src="https://img.shields.io/badge/openai-0.27.0-blue" /> <img src="https://img.shields.io/badge/pandas-2.1.3-yellow" /> <img src="https://img.shields.io/badge/opencv--python-4.5.5.64-red" /> <img src="https://img.shields.io/badge/python--dotenv-0.21.1-green" /> </p>

1
## Table of Contents
1. [Introduction](#introduction)
2. [Report](#report)
   1. [Motivation](#motivation)
   2. [Process](#process)
   3. [Conclusion & Outline](#conclusion-&-outline)
3. [Pipeline](#pipeline)
   1. [Features](#features)
   2. [Project Structure](#project-structure)
   3. [Setup Instructions](#setup-instructions)
      1. [Clone the Repository](#clone-the-repository)
      2. [Configuration](#configuration)
4. [Usage](#usage)
5. [Questionnaire Sources](#questionnaire-sources)
6. [Troubleshooting](#troubleshooting)
7. [License](#license)
8. [References](#references)




## 1. Introduction

**GPT UX Ratings** is designed to evaluate how different traffic situations may be perceived by passengers in autonomous vehicles. The system uses OpenAI's GPT model to simulate answers to established UX rating questions.

This method helps generate synthetic datasets for traffic safety, automation research, and behavioral studies involving passenger perceptions.

## 2.  Report

### 1. Motivation 
Automated vehicles will increasingly need to understand how their drivers experience different traffic situations in order to support them appropriately. Today, assessing **mental workload, trust, or perceived safety** still relies on intrusive sensors, controlled lab environments, or frequent self reports. These methods produce valuable data, but they do not scale to everyday driving and cannot provide continuous feedback without interrupting the user.

This project explores a different approach: instead of measuring the driver directly, it focuses on what can be **inferred from the scene itself.** The pipeline takes driving videos, uses OpenAI models to generate structured descriptions of what is happening, and then uses these descriptions to simulate how people might respond on established questionnaires that capture workload, trust, and situational awareness. An important aspect of this work is to examine whether large language models have enough scene understanding and contextual reasoning to approximate such subjective ratings at all.

The idea behind the pipeline is to explore whether a scene based, non intrusive method could contribute to estimating driver state in the future and to test how far current LLM capabilities can already take us.

### 2. Process 

#### First steps and general idea

- **Video collection:**  
  Driving-scene videos were sourced from YouTube, focusing on a wide range of real-world events, including near-accidents, collisions, varying weather conditions, and diverse road environments.

- **Field-of-view cropping:**  
  Each video was cropped to display only the **forward windshield view**, removing irrelevant elements outside the driver’s visual field.

- **Duration trimming:**  
  Videos were clipped to a standardized length between **7 and 30 seconds** to ensure consistency across the dataset.

- **Quality filtering:**  
  The initial set of 89 videos was reduced after discarding clips that did not meet the required quality or content criteria.

- **Pipeline construction:**  
At first the idea was to send the **full video clip** to get ratings for simulated participants, but we looked into how **video processing** is done through chatgpt and since it only considers a **certain number of frames**, in order to have more control over the what frames were taken into account we decided to **extract frames as a separate process** and then send them to the model. However, the model has an **upper limit for images** that can be sent in one prompt, that is not compatible with the **amount of detail** that is needed to analyze **multi-second video**. We started with **4 frames per second** but increased this to **8 frames per second** in order to try and enhance the **qualities of the summaries.** This is why we decided to first use the model to **generate summaries for subsequent batches** and **combine them to keep a coherent story** of whats going on in the driving scene and to send that to the model to obtain the ratings.


**Prompt Development**

The initial prompt was designed with only the **basic information** thought necessary to generate scene summaries. However, through **trial and error**, we iteratively refined the prompt to improve the **accuracy, clarity, and relevance** of the summaries. This process ensured that the GPT-based model produced outputs better aligned with the events depicted in the driving videos.

This was an earlier version: 

_Generate a summary for the following frames extracted from a driving video at the rate of one frame every quarter second. 
You are seeing the view through a windshield of an automated vehicle. 
Build on the previous summary without repeating what was already established unless it is necessary for continuity. Focus on new or changing details — moving objects, traffic signals, pedestrians, and notable events.
Keep the style factual and objective, avoiding poetic or overly descriptive language.
Use 1–2 clear sentences per update, enough to capture what is happening without overexplaining. If nothing significant changes, summarize that briefly in one short sentence. Respond only with the update. Here is the summary so far: {summary_so_far}_


It was then iteratively updated: 

_Generate a summary for the following frames extracted from a driving video at the rate of eight frames per second. **They are labeled in order with a number in the top left corner.** You are seeing the view through a windshield of an automated vehicle. Build on the provided summary of previous             frames without repeating what was already established unless it is necessary for continuity. Focus on new, unexpected, or changing details — moving objects, traffic signals, pedestrians, and notable events. **Ignore any subtitles or encoded time or speed information but consider that every frame you       see is 1/8 of a second apart.**
Keep the style factual and objective, avoid poetic or overly descriptive language.
Use 1–2 clear sentences per update, enough to capture what is happening without overexplaining. **If what you see adds information to or contradicts something previously stated in the summary, point it out.** Respond only with the update. Here is the summary so far: {summary_so_far}_

These changes were made to solve the following issues: 
- some summaries raised suspicions that the **frames within a batch were not procressed in the correct order**
- some video clips included the **car's speed** or **subtitles of passengers' speech** which were mentioned in the summaries
- we hoped that by putting emphasis on the **framerate** the summaries would more accurately reflect the **speed of things happening in the scene**
- we wanted the summaries to be **more coherent** and for the model to able to **integrate new information with the previously seen frames**  

#### Project Timeline (after video collection)

![Project Timeline](materials/timeline.svg)


### 3. Conclusion & Outlook
The project shows that, in its current form, **GPT-5.1** is not yet reliable enough to generate meaningful **synthetic UX ratings** for **complex driving scenarios.** While the model was able to recognize many scene elements and maintain a degree of narrative continuity, it frequently **overlooked or misinterpreted safety-critical details.** This resulted in summaries that were sometimes coherent at the micro-level but incomplete at the event-level, especially in situations involving **multi-frame dynamics** such as near-misses, abrupt maneuvers, or collisions. Another frequently occurring issue was that **lane boundaries were interpreted incorrectly.** Small details that were irrelevant to the evolving situation were often described in detail, showing that although the model is capable of identifying components of the scene relatively reliably, their relevance and interaction are not captured.

Even with careful prompt engineering, higher frame rates, and explicit instructions to correct earlier statements, the model struggled with **temporal reasoning.** When relevant cues unfolded across several frames, **early omissions propagated** throughout the summary, preventing the model from reconstructing the full situation. This directly affected the subsequent rating generation: the synthetic participants often failed to map the scenario to appropriate **UX constructs** (e.g., **perceived safety, predictability, mental workload**), occasionally assuming that no crash or dangerous event occurred despite clear visual evidence.

The ratings themselves further highlighted these limitations. Several responses contained **out-of-range values, mismatches** between justifications and events, or **unrealistically low standard deviations**—a well-known issue in LLM-based social simulations. At times, the outputs did not conform to the intended questionnaire scales at all. These issues demonstrate that, without additional constraints, LLMs tend to produce overly consistent, insufficiently varied, and sometimes **semantically misaligned synthetic data.**

Overall, the findings emphasize the need for **systematic post-processing, quality validation,** and more **robust prompting structures** when attempting to use LLMs as proxies for human UX ratings in dynamic, real-world environments. While the model shows promise in basic scene description, it currently lacks the **temporal precision** and **psychological grounding** required to generate trustworthy UX assessments for automated driving research.

In the future, this pipeline could be tested with other models better suited for vision and scene description. For example, a **Llava model** could be used and the **API key** will need to be changed in the .env file and the [**API function call**](summaries_ratings#L193) in the ratings script will need to be updated. Such models may offer stronger temporal coherence, improved detection of events, and more reliable interpretation of subtle scene cues. It would be possible to determine whether these variations result in more reliable summaries and more significant UX scores by evaluating them inside the same pipeline.  

Once alternative vision models are integrated, if the summaries are deemed accurate, it becomes essential to validate the resulting rating by comparing the **synthetic ratings** to **real human responses**, to ensure that improvements in scene understanding translate into meaningful UX assessments. **Benchmarking** the model's assessments against real human data is the next step to determine where the model differs the most. These contrasts may show whether the LLM tends to misinterpret events. By systematically comparing synthetic ratings across demographic groups or driving experiences and assessing whether these patterns correspond with known findings, potential **biases** can be identified.


## 3. Pipeline 
### 1. Features 
This project uses **OpenAI’s GPT-5** model to analyze frames extracted from driving videos and simulate human emotional reactions (as CSV ratings) for different age groups and gender.

It automates the following:
1. Extracts frames every **¼ second** from each video.
2. Sends the frames in batches of 10 in  **GPT-5** API.
3. Prompts the model to generate a summary of the videos events  
4. Prompts the model to generate a participant rating of the situations based on the summaries 
5. Saves all responses as **CSV files**.


### 2. Project Structure

```plaintext

GPT_UX_ratings/
│
├── README.md
├── requirements.txt
├── .gitignore
├── frames
├── summaries_ratings
│
├── materials/                       
|   ├── ratings/
│   ├── summaries/
│   ├── videos/
│   ├── In-Car Video Spread Sheet - Sheets…
│   └── timeline.svg

```

### 4. Setup Instructions

#### 1. Clone the Repository
Clone the repository and install dependencies:

```bash
git clone https://github.com/arcymonka/GPT_UX_ratings.git
cd GPT_UX_ratings
pip install -r requirements.txt
```
#### 2. Configuration

Create a `.env` file in the root directory to configure environment variables:

```ini
OPENAI_API_KEY="your_openai_key"
VIDEO_PATH="path/to/videos"
OUTPUT_PATH="path/to/frames"
SUMMARY_PATH="path/to/summaries"
RATINGS_OUTPUT_PATH="path/to/output/ratings"
RANDOM_SEED=42
```


Ensure all referenced directories exist and contain valid data files (e.g., `.txt` summaries in `SUMMARY_PATH`).


## 4. Usage

Run the scripts in the following order: 
```bash 
python frames.py
python part_rat.py
```


By default, the scripts will:

- Load image frames from the `frames/` directory.  
- Process each frame using the OpenAI API.  
- Read each summary file from the configured `SUMMARY_PATH`
- Generate a prompt for each age/gender combination
- Call OpenAI's API to simulate ratings
- Save each response to a CSV in `RATINGS_OUTPUT_PATH`


## 5. Questionnaire Sources

The 30-item rating scale is built upon validated measures from several academic sources:

- **Perceived Safety** – Faas et al. (2020)
- **Trust in Automation** – Körber (2019)
- **Predictability** – Körber (2019)
- **Acceptance** – Van der Laan et al. (1997)
- **SART (Situation Awareness Rating Technique)** – Taylor (2017)

## 6. Troubleshooting

| Issue                             | Possible Cause                                | Solution                                   |
|----------------------------------|-----------------------------------------------|--------------------------------------------|
| No CSV output                    | API key issue or missing input files          | Check `.env` values and summary folder     |
| Ratings are misformatted         | Unexpected model output                       | Review prompt and ensure GPT model validity|
| API error or rate limit exceeded | Too many requests or invalid model version    | Try again later or adjust request volume or add money to API account |
| Output folder not created        | Missing permissions or invalid path           | Ensure script can create/write to paths    |

## 7. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 8. References 

Anthis, J. R., Liu, R., Richardson, S. M., Kozlowski, A. C., Koch, B., Brynjolfsson, E., Evans, J., & Bernstein, M. S. (2025). LLM social simulations are a promising research method (arXiv preprint arXiv:2504.02234 v2). arXiv. https://doi.org/10.48550/arXiv.2504.02234

Faas, S. M., Mattes, S., Kao, A. C., & Baumann, M. (2020). Efficient paradigm to measure street-crossing onset time of pedestrians in video-based interactions with vehicles. Information, 11(7), 360. https://doi.org/10.3390/info11070360

Körber, M. (2019). Theoretical considerations and development of a questionnaire to measure trust in automation. In S. Bagnara, R. Tartaglia, S. Albolino, T. Alexander, & Y. Fujita (Eds.), Proceedings of the 20th Congress of the International Ergonomics Association (IEA 2018) (Advances in Intelligent Systems and Computing, Vol. 823, pp. 13-30). Springer. https://doi.org/10.1007/978-3-319-96074-6_2

Taylor, R. M. (2017). Situational Awareness Rating Technique (SART): The Development of a Tool for Aircrew Systems Design. In Situational Awareness (pp. 111–128). Routledge. https://doi.org/10.4324/9781315087924-8

Van Der Laan, J. D., Heino, A., & De Waard, D. (1997). A simple procedure for the assessment of acceptance of advanced transport telematics. Transportation Research Part C: Emerging Technologies, 5(1), 1–10. https://doi.org/10.1016/S0968-090X(96)00025-3
