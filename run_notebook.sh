#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running notebook execution script from: $(pwd)"
# Git config (set once per repo) – adjust as needed
if ! git config user.email >/dev/null; then
  git config --local user.email "rois.khoiron@gmail.com"
  git config --local user.name "roiskhoiron"
fi

echo "Git user configured: $(git config user.name) <$(git config user.email)>"
# Activate virtual environment (optional – skip if not present)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# Directory where notebooks live
NOTEBOOK_DIR="notebooks"

echo "Installing dependencies...[1/6]"
pip install -r requirements.txt
pip install kagglehub tensorflowjs

echo "Executing notebooks 01...[2/6]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/01_data_ingestion.ipynb" --output "01_data_ingestion.ipynb"

echo "Executing notebooks 02...[3/6]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/02_preprocessing.ipynb" --output "02_preprocessing.ipynb"

echo "Executing notebooks 03...[4/6]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/03_model_training.ipynb" --output "03_model_training.ipynb"

echo "Executing notebooks 04...[5/6]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/04_evaluation.ipynb" --output "04_evaluation.ipynb"

echo "All notebooks executed successfully."

echo "Cleaning up any temporary files...[6/6]"
# Commit executed notebooks and push to origin master
git add .
git commit -m "chore: executed notebooks" || true
git push origin master
