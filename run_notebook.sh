#!/usr/bin/env bash
set -euo pipefail

# Git config (set once per repo) – adjust as needed
if ! git config user.email >/dev/null; then
  git config --local user.email "rois.khoiron@gmail.com"
  git config --local user.name "roiskhoiron"
fi

# Activate virtual environment (optional – skip if not present)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Directory where notebooks live
NOTEBOOK_DIR="notebooks"

# setup environment
pip install -r requirements.txt
pip install --no-deps kagglehub && pip install tensorflowjs

# Ensure kernel uses correct python
python -m ipykernel install --user --name .venv312 --display-name "Python 3.14 (.venv)" --force

# Execute notebooks in order
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/01_data_ingestion.ipynb" --output "${NOTEBOOK_DIR}/01_data_ingestion.ipynb"

jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/02_preprocessing.ipynb" --output "${NOTEBOOK_DIR}/02_preprocessing.ipynb"

jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/03_model_training.ipynb" --output "${NOTEBOOK_DIR}/03_model_training.ipynb"

jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/04_evaluation.ipynb" --output "${NOTEBOOK_DIR}/04_evaluation.ipynb"

echo "All notebooks executed successfully."

# Commit executed notebooks and push to origin master
git add .
git commit -m "chore: executed notebooks" || true
git push origin master
