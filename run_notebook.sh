#!/usr/bin/env bash
set -euo pipefail

# Git config (set once per repo) – adjust as needed
if ! git config user.email >/dev/null; then
  git config --local user.email "rois.khoiron@gmail.com"
  git config --local user.name "roiskhoiron"
fi

# Activate virtual environment if needed (optional)
# source .venv/bin/activate

# Directory where notebooks live
NOTEBOOK_DIR="$(pwd)/notebooks"

# setup environment
pip install -r requirements.txt
pip install tensorflowjs kagglehub

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