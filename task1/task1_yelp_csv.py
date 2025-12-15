import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.0-flash-lite"

def call_gemini_json(prompt: str) -> dict:
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )
    text = response.text
    try:
        return json.loads(text)
    except Exception:
        text_clean = text.strip().strip("`")
        try:
            return json.loads(text_clean)
        except Exception:
            return {"raw_response": text_clean}

df = pd.read_excel("/content/yelp_test.xlsx")
df = df[["text", "stars"]].dropna()

sample_size = 20
df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
df_sample.head()

def prompt_p1(review: str) -> str:
    return f"""
You are an expert sentiment annotator for restaurant reviews.

Task:
Given a Yelp review, assign an integer star rating from 1 to 5 that best reflects the customer's overall experience.

Rating rules:
- 1 star: Very bad experience, strong negative language, no redeeming aspects.
- 2 stars: Mostly negative, some minor positive points but overall unhappy.
- 3 stars: Mixed/neutral, both good and bad points, overall average.
- 4 stars: Mostly positive, minor issues but customer is generally happy.
- 5 stars: Excellent experience, strong positive language, would highly recommend.

Instructions:
1. Read the review carefully.
2. Choose exactly one integer rating from 1, 2, 3, 4, or 5.
3. Return a valid JSON object with this exact schema and nothing else:

{{
  "predicted_stars": <integer 1-5>,
  "explanation": "<brief explanation of why this rating fits the review>"
}}

Review:
\"\"\"{review}\"\"\"
"""

def prompt_p2(review: str) -> str:
    return f"""
You are rating Yelp restaurant reviews on a 1–5 star scale.

Step-by-step process:
1. Briefly identify key positive signals (e.g., good food, friendly staff, fast service).
2. Briefly identify key negative signals (e.g., rude staff, long wait, bad food, dirty place).
3. Decide the final rating using this rubric:
   - 1 star: Strong negative tone, major problems, no redeeming features.
   - 2 stars: Mostly negative, some minor positives but experience is bad overall.
   - 3 stars: Balanced or neutral, mix of positives and negatives.
   - 4 stars: Mostly positive, small issues but overall good experience.
   - 5 stars: Strongly positive, enthusiastic praise, would recommend to others.

Requirements:
- Choose exactly one integer rating from 1, 2, 3, 4, or 5.
- Think step by step internally, then only output the final JSON.

Output:
Return a valid JSON object ONLY, with this schema:

{{
  "predicted_stars": <integer 1-5>,
  "explanation": "<1-2 sentence explanation based on the signals and rubric>"
}}

Review:
\"\"\"{review}\"\"\"
"""

FEW_SHOT_EXAMPLES = [
    {
        "review": "The food was terrible, service was slow, and the place was dirty. I will never come back.",
        "stars": 1,
        "why": "Strong negative tone and multiple serious issues."
    },
    {
        "review": "Burger was okay and fries were decent, but the service took a while. Overall it was fine.",
        "stars": 3,
        "why": "Mixed experience with both positives and negatives."
    },
    {
        "review": "Amazing sushi, super friendly staff, and quick service. Highly recommended!",
        "stars": 5,
        "why": "Strong positive language and clear satisfaction."
    },
]

def prompt_p3(review: str) -> str:
    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_str += f"""
Review: \"\"\"{ex['review']}\"\"\"
Rating: {ex['stars']} stars
Reason: {ex['why']}
"""
    return f"""
You are classifying Yelp reviews into star ratings from 1 to 5.

Here are some examples:

{examples_str}

Now, rate the new review in the same style.

Instructions:
- Use only integer ratings 1, 2, 3, 4, or 5.
- Follow the pattern shown in the examples.
- Return ONLY a valid JSON object with this schema:

{{
  "predicted_stars": <integer 1-5>,
  "explanation": "<brief explanation, similar to the example reasons>"
}}

New review:
\"\"\"{review}\"\"\"
"""

def run_experiment(df_sample: pd.DataFrame, prompt_fn, approach_name: str) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        review_text = row["text"]
        true_stars = int(row["stars"])

        prompt = prompt_fn(review_text)
        result = call_gemini_json(prompt)

        raw_json = json.dumps(result, ensure_ascii=False)

        pred = None
        explanation = None
        if isinstance(result, dict):
            pred = result.get("predicted_stars")
            explanation = result.get("explanation")

        try:
            if pred is not None:
                pred = int(pred)
        except Exception:
            pred = None

        rows.append({
            "approach": approach_name,
            "text": review_text,
            "true_stars": true_stars,
            "predicted_stars": pred,
            "explanation": explanation,
            "raw_json": raw_json,
        })
    return pd.DataFrame(rows)

df_p1 = run_experiment(df_sample, prompt_p1, "P1_simple")
df_p2 = run_experiment(df_sample, prompt_p2, "P2_cot_rubric")
df_p3 = run_experiment(df_sample, prompt_p3, "P3_few_shot")

df_all = pd.concat([df_p1, df_p2, df_p3], ignore_index=True)
df_all.to_csv("task1_predictions.csv", index=False)
df_all.head()

metrics = []

for approach, df_group in df_all.groupby("approach"):
    mask_valid = df_group["predicted_stars"].apply(lambda x: isinstance(x, int) and 1 <= x <= 5)
    df_valid = df_group[mask_valid]

    if len(df_valid) > 0:
        acc = accuracy_score(df_valid["true_stars"], df_valid["predicted_stars"])
    else:
        acc = 0.0

    json_valid_rate = df_group["raw_json"].notna().mean()
    consistency_rate = mask_valid.mean()

    metrics.append({
        "approach": approach,
        "n_samples": len(df_group),
        "n_valid_predictions": len(df_valid),
        "accuracy": acc,
        "json_valid_rate": json_valid_rate,
        "consistency_1_to_5": consistency_rate,
    })

metrics_df = pd.DataFrame(metrics)
metrics_df