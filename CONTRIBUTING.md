# Contributing

Thanks for your interest in improving this project!

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Quality checks
```bash
ruff check .
black .
pytest -q
```

## Pull Requests
- Keep PRs small and focused.
- Add or update tests where it makes sense.
- Update the README if behavior changes.

