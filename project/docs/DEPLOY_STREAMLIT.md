# Deploying the Streamlit app (Community Cloud)

This repo ships with a ready-to-deploy Streamlit app (`app.py`).

## Prerequisites
- Your code is pushed to a public GitHub repo (free tier)
- `requirements.txt` exists in the repo root (Streamlit Cloud uses it)

## Steps
1. Go to Streamlit Community Cloud
2. Click **Create app**
3. Select your repo + branch
4. Set:
   - **Main file path**: `project/app.py`
5. Click **Deploy**

## Common issues

### "streamlit is not recognized" (local)
Install requirements:
```bash
pip install -r requirements.txt
```

## After deploy
Copy your app URL and replace the **Deploy** badge link at the top of `README.md`.
