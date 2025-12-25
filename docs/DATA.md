# Data Notes

## Default dataset
The default configuration downloads a Lending Club dataset via KaggleHub.

### Setup Kaggle credentials
Follow Kaggle's API instructions to create a token, then place it at:
- `~/.kaggle/kaggle.json` (Mac/Linux)
- `%USERPROFILE%\.kaggle\kaggle.json` (Windows)

## Offline / reproducible runs
Set:
- `LENDING_CLUB_CSV=/path/to/your.csv`

## Sample input
A small feature-only CSV is included at `data/sample_input.csv` for testing the Streamlit app.

