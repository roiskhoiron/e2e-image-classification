#!/usr/bin/env bash
set -euo pipefail

# Git config (set once per repo) – adjust as needed
if ! git config user.email >/dev/null; then
  git config --local user.email "you@example.com"
  git config --local user.name "Your Name"
fi

# Activate virtual environment if needed (optional)
# source .venv/bin/activate

# Directory where notebooks live
NOTEBOOK_DIR="$(pwd)/notebooks"

# Execute notebooks in order
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/01_data_ingestion.ipynb" --output "${NOTEBOOK_DIR}/01_data_ingestion_executed.ipynb"

jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/02_preprocessing.ipynb" --output "${NOTEBOOK_DIR}/02_preprocessing_executed.ipynb"

jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/03_model_training.ipynb" --output "${NOTEBOOK_DIR}/03_model_training_executed.ipynb"

jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/04_evaluation.ipynb" --output "${NOTEBOOK_DIR}/04_evaluation_executed.ipynb"

echo "All notebooks executed successfully."

# Commit executed notebooks and push to origin master
git add "${NOTEBOOK_DIR}"/*_executed.ipynb
git commit -m "chore: executed notebooks" || true
git push origin master