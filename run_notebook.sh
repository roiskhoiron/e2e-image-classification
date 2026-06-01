echo "Running notebook execution script..."
#!/usr/bin/env bash
set -euo pipefail

echo "Setting up environment and executing notebooks..."
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
echo "Notebooks will be executed from: ${NOTEBOOK_DIR} [1/4]"
NOTEBOOK_DIR="notebooks"

echo "Installing dependencies... [2/4]"
# setup environment
pip install -r requirements.txt
pip install kagglehub tensorflowjs

echo "Dependencies installed. Executing notebooks... [3.1/4]"
# Execute notebooks in order
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/01_data_ingestion.ipynb" --output "${NOTEBOOK_DIR}/01_data_ingestion.ipynb"

echo "Executed 01_data_ingestion.ipynb successfully. Continuing with next notebooks... [3.2/4]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/02_preprocessing.ipynb" --output "${NOTEBOOK_DIR}/02_preprocessing.ipynb"

echo "Executed 02_preprocessing.ipynb successfully. Continuing with next notebooks... [3.3/4]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/03_model_training.ipynb" --output "${NOTEBOOK_DIR}/03_model_training.ipynb"

echo "Executed 03_model_training.ipynb successfully. Continuing with next notebooks... [3.4/4]"
jupyter nbconvert --to notebook --execute \
  "${NOTEBOOK_DIR}/04_evaluation.ipynb" --output "${NOTEBOOK_DIR}/04_evaluation.ipynb"

echo "All notebooks executed successfully."

echo "Committing executed notebooks and pushing to origin master... [4/4]"
# Commit executed notebooks and push to origin master
git add .
git commit -m "chore: executed notebooks" || true
git push origin master
