import os
import datetime
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
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
    """Load existing feedback from CSV, or create empty DataFrame."""
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
    """Persist feedback to CSV."""
    df.to_csv(DATA_PATH, index=False)


# ---------- GEMINI CALL ----------
def generate_user_response(rating: int, review: str) -> str:
    """Generate a polite user-facing response based on rating and review."""
    prompt = f"""
You are a customer support assistant for a restaurant.

User rating: {rating} stars
User review: \"\"\"{review}\"\"\"

Write a short, polite response:
- Acknowledge their experience.
- If rating <= 3, apologize and show willingness to improve.
- If rating >= 4, thank them and invite them back.
Keep it under 4 sentences.
"""
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return (response.text or "").strip()


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="User Feedback", page_icon="⭐")
st.title("Feedback – User Dashboard")

st.write("Share your experience by choosing a rating and writing a short review.")

rating = st.selectbox("Your rating (1–5 stars)", [1, 2, 3, 4, 5])
review = st.text_area("Your review")

if st.button("Submit"):
    if not review.strip():
        st.warning("Please write a review before submitting.")
    else:
        with st.spinner("Generating response..."):
            ai_response = generate_user_response(rating, review)

        df = load_data()
        new_row = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "rating": rating,
            "review": review,
            "ai_response": ai_response,
            "ai_summary": "",
            "ai_actions": "",
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)

        st.success("Thank you for your feedback!")
        st.subheader("Our response")
        st.write(ai_response)
