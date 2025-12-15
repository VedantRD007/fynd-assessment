text
# Fynd AI Intern - Take Home Assessment

AI-powered feedback system with Yelp rating prediction and dual dashboards. Built with Streamlit, Gemini 2.0 Flash, and Pandas.

## 📋 Table of Contents

- [Task 1: Rating Prediction](#task-1-rating-prediction)
- [Task 2: Two-Dashboard System](#task-2-two-dashboard-system)
- [Deployments](#deployments)
- [Local Setup](#local-setup)
- [Tech Stack](#tech-stack)

## Task 1: Rating Prediction

**Notebook**: `task1/task1_notebook.ipynb`

Evaluated 3 prompting strategies on Yelp reviews dataset (~20 samples due to API limits):

| Approach | Accuracy | JSON Validity | Consistency (1-5) |
|----------|----------|---------------|-------------------|
| P1 Simple | 60% | 1.0| 1.0 |
| P2 CoT+Rubric | 63% | 1.0 | 1.0 |
| P3 Few-shot | 66% | 1.0 | 1.0 |

**Key findings**: Few-shot prompting showed highest accuracy and consistency.

## Task 2: Two-Dashboard System

**User Dashboard** (`task2/app_user.py`):
- Select 1-5 star rating
- Write review
- Get AI response via Gemini
- Stores to shared `feedback.csv`

**Admin Dashboard** (`task2/app_admin.py`):
- Live view of all submissions
- AI-generated summaries & recommended actions
- Analytics (avg rating, rating distribution)

Both share `feedback.csv` as persistent storage.

## 🚀 Deployments

| Dashboard | URL |
|-----------|-----|
| User | [https://your-user-app.streamlit.app](https://fynd-assessment-mmmkgbkip9jaunuvy92368.streamlit.app/) |
| Admin | [https://your-admin-app.streamlit.app](https://fynd-assessment-erzqsrtxpnazwvyw4sd87q.streamlit.app/) |

## 🛠️ Local Setup

Clone repo
git clone https://github.com/yourusername/fynd-assessment.git
cd fynd-assessment/task2

Create virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/Mac

.venv\Scripts\activate # Windows
Install dependencies
pip install -r requirements.txt

Set API key
export GEMINI_API_KEY="your-key-here"

Run apps
streamlit run app_user.py # Terminal 1
streamlit run app_admin.py # Terminal 2

text

## Tech Stack

Frontend: Streamlit
Backend: Python, Pandas
LLM: Google Gemini 2.0 Flash Lite (free tier)
Storage: CSV (feedback.csv)
Deployment: Streamlit Community Cloud

text

## 📄 Report

[Short Report PDF](https://github.com/yourusername/fynd-assessment/blob/main/report/report.pdf)

**Approach Summary**:
- Task 1: Implemented 3 prompting strategies (zero-shot, CoT, few-shot) with structured JSON evaluation
- Task 2: Built full-stack web app with shared CSV storage, LLM-powered responses/summaries/actions
- Design decisions prioritized simplicity, free-tier compatibility, and rapid deployment

## Limitations & Future Work

- Used 20 samples for Task 1 due to Gemini free-tier limits (noted in notebook)
- CSV storage works for demo; production would use PostgreSQL/Supabase
- Could add user authentication for admin dashboard

---

⭐ **Built for Fynd AI Intern Assessment**  
📅 December 2025  
👨‍💻 Vedant [VedantRD007]
