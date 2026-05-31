# End-to-End Image Classification Portfolio

This repository implements an end-to-end image classification pipeline using TensorFlow/Keras as a portfolio project.

## Project Structure

```
project_root/
├── notebooks/
│   ├── 01_data_ingestion.ipynb     # Data download, validation, and splitting
│   ├── 02_preprocessing.ipynb      # tf.data pipeline with augmentation
│   ├── 03_model_training.ipynb     # CNN model definition and training
│   └── 04_evaluation.ipynb         # Evaluation: curves, confusion matrix, inference
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_builder.py      # Utilities for scanning, validating, and splitting images
│   └── pipeline/
│       ├── __init__.py
│       ├── data_pipeline.py        # tf.data pipeline construction
│       ├── model.py                # CNN model building and training
│       ├── evaluation.py           # Plotting and metrics
│       └── export_utils.py         # Export to SavedModel, TFLite, TFJS
├── data/
│   ├── raw/                        # Raw downloaded images
│   └── processed/                  # Train/val/test split images
├── models_output/
│   ├── checkpoints/                # Best model weights (.keras)
│   ├── saved_model/                # TensorFlow SavedModel format
│   ├── tflite/                     # TensorFlow Lite (.tflite + labels.txt)
│   └── tfjs/                       # TensorFlow.js model.json + weight files
├── reports/
│   └── figures/                    # Plots: accuracy/loss curves, confusion matrix
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Pipeline Overview

1. **Data Ingestion** – Download dataset, validate images, split into train/val/test (80/10/10).
2. **Preprocessing** – Build `tf.data.Dataset` with dynamic resizing, rescaling to [0,1], isolated augmentation (training only), caching, and prefetching.
3. **Model Training** – Train a CNN (Conv2D blocks → Flatten → Dense → Dropout → Softmax) with Adam optimizer, EarlyStopping, and ModelCheckpoint.
4. **Evaluation** – Generate accuracy/loss curves, confusion matrix, and classification report on the test set.
5. **Export** – Convert the trained model to SavedModel, TFLite (with quantization), and TFJS for deployment.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebooks in order:
   - `notebooks/01_data_ingestion.ipynb`
   - `notebooks/02_preprocessing.ipynb`
   - `notebooks/03_model_training.ipynb`
   - `notebooks/04_evaluation.ipynb`

   Each notebook can be executed independently as they import the necessary functions from the `src/` package.

3. To export the model after training, run the export script or use the functions in `src/pipeline/export_utils.py`.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (for image validation)
- Matplotlib, scikit-learn (for evaluation)
- Jupyter (for notebooks)
- TensorFlow.js converter (for TFJS export)

See `requirements.txt` for exact versions.

## Notes

- The project is designed to handle >10,000 images with varying resolutions without memory issues by using `tf.data` streaming.
- Data augmentation is applied only to the training set to prevent data leakage.
- All exported artifacts (SavedModel, TFLite, TFJS) are verified to load and produce consistent predictions.
