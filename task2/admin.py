import os
import json
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Make sure GEMINI_API_KEY is set in your environment / hosting platform
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("API_KEY not found. Please set GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-lite"
DATA_PATH = "feedback.csv"


# ---------- DATA HELPERS ----------
def load_data():
    """Load feedback data from CSV or return empty DataFrame."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(
        columns=[
            "timestamp",
            "rating",
            "review",
            "ai_response",
            "ai_summary",
            "ai_actions",
        ]
    )


def save_data(df: pd.DataFrame):
    """Save feedback data to CSV."""
    df.to_csv(DATA_PATH, index=False)


# ---------- GEMINI CALL ----------
def generate_summary_and_actions(rating, review):
    """
    Use Gemini to produce:
    - a short summary of the review
    - 2–3 recommended actions
    """
    prompt = f"""
You are an operations analyst for a restaurant.

Given this customer feedback, produce:
1) A one-sentence summary of the feedback.
2) 2–3 concrete recommended actions for the team.

User rating: {rating} stars
User review: \"\"\"{review}\"\"\"

Return JSON only, with:
{{
  "summary": "<short summary>",
  "actions": ["<action1>", "<action2>", "<action3>"]
}}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )

    raw = (response.text or "").strip()

    try:
        # If model wraps JSON in ```json ... ``` remove backticks
        raw_clean = raw.strip("`")
        data = json.loads(raw_clean)
        summary = data.get("summary", "")
        actions_list = data.get("actions", [])
        if isinstance(actions_list, list):
            actions_text = "; ".join(str(a) for a in actions_list)
        else:
            actions_text = str(actions_list)
    except Exception as e:
        # Show error and raw output so you can see what's happening
        st.error(f"JSON parse error: {e}")
        st.write("Raw response from Gemini:", raw)
        summary = ""
        actions_text = ""

    return summary, actions_text


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Admin Feedback Dashboard", page_icon="📊")
st.title("Feedback – Admin Dashboard")

df = load_data()

if len(df) == 0:
    st.info("No feedback yet. Ask users to submit reviews on the User Dashboard.")
else:
    # Fill missing summaries/actions using Gemini
    needs_update = df["ai_summary"].isna() | (df["ai_summary"] == "")
    updated = False

    for idx in df[needs_update].index:
        rating = df.loc[idx, "rating"]
        review = df.loc[idx, "review"]

        # Debug: show which row is being processed
        st.write(f"Processing row index: {idx}")

        if not isinstance(review, str) or not review.strip():
            continue

        with st.spinner(f"Generating summary/actions for row {idx}..."):
            summary, actions = generate_summary_and_actions(rating, review)

        df.loc[idx, "ai_summary"] = summary
        df.loc[idx, "ai_actions"] = actions
        updated = True

    if updated:
        save_data(df)
        st.success("Updated summaries and actions saved.")

    st.subheader("All submissions")
    st.dataframe(
        df[["timestamp", "rating", "review", "ai_response", "ai_summary", "ai_actions"]],
        use_container_width=True,
    )

    st.subheader("Analytics")
    st.write(f"Total submissions: {len(df)}")

    if "rating" in df.columns and len(df["rating"]) > 0:
        try:
            avg_rating = df["rating"].astype(float).mean()
            st.write(f"Average rating: {avg_rating:.2f}")
        except Exception:
            st.write("Average rating: N/A")

        rating_counts = df["rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)
    else:
        st.write("No rating data available yet.")
