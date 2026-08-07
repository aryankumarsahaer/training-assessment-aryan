# Coding & Logic Assessment - Eternal Robotics Training

This repository contains the clean, fully-optimized implementations for the internship training assessment.

---

## Tasks Overview

### 1. Pattern Generation (`task_pattern.py`)
Generates a dynamic, symmetrical arrow star pattern for any odd integer size $N \geq 5$.

### 2. Telemetry Logger (`task_logger.py`)
A continuous system diagnostics utility that logs CPU usage %, Memory usage, and Disk usage to a JSON file (`telemetry_log.json`) at regular intervals.

### 3. ASCII Reduction (`task_ascii.py`)
Converts a name string into its ASCII character values, sums them, and recursively reduces the sum to a single-digit result.

### 4. GPU Image Classification & Optimization (`task_model.py`)
An automated machine learning pipeline built specifically for classification tasks.

---

## Task 4: Detailed Implementation Details

The model training pipeline is composed of four main stages:

### A. Dynamic Bounding Box Cropping
Before training the classifier, the script reads annotation bounding boxes from the target dataset:
1. **Coordinate Conversion**: YOLO normalized bounding box format (`class_id x_center y_center width height`) is converted into absolute pixel coordinate boundaries (`x1`, `y1`, `x2`, `y2`) based on the input image dimensions.
2. **Crop Generation**: Sub-images containing the objects of interest are cropped from the original scanned images.
3. **Data Splitting**: Crops are split into an 80/20 train/validation ratio. If a class contains only a single image crop, the script automatically duplicates it into both splits to satisfy YOLOv8's training requirements.
4. **Dataset Directory Structure**: The cropped images are saved in:
   ```text
   classifier_dataset/
   ├── train/
   │   ├── QR/
   │   ├── qr/
   │   ├── cosgign_no/
   │   └── reg_no/
   └── val/
       ├── QR/
       ├── qr/
       ├── cosgign_no/
       └── reg_no/
   ```

### B. Hyperparameter Optimization with Optuna
The pipeline integrates the **Optuna** framework to search for the best hyperparameter configuration:
* **Learning Rate (`lr0`)**: Searched using a log-uniform float distribution between `1e-4` and `1e-2`.
* **Optimizer Choice**: Evaluates the best performance between `'Adam'`, `'AdamW'`, and `'SGD'`.
* **Backbone Layer Freezing (`freeze`)**: Evaluates freezing between `0` and `8` layers. It is capped at `8` layers to prevent freezing the classification head (layer 9), which would result in a zero trainable parameters error.
* **Trial Objective**: Each trial runs classification training for 1 epoch, returning the validation Top-1 accuracy to Optuna to determine the optimal configuration.

### C. Output Directory Structure (`runs/classify/`)
YOLOv8 stores training runs in the `runs/classify/` folder. Every individual training invocation automatically increments the folder name to prevent overwriting past results:
* `train` through `train-7`: Initial check runs.
* `train-8` (Trial 0): Evaluated hyperparameter set 1.
* `train-9` (Trial 1): Evaluated hyperparameter set 2.
* `train-10` (Trial 2): Evaluated hyperparameter set 3.
* `train-11` (Trial 3): Evaluated hyperparameter set 4.
* `train-12` (Trial 4): Evaluated hyperparameter set 5.
* **`train-13` (Final Run)**: The training run executed after Optuna completed, using the best discovered hyperparameters.

### D. Training Outputs & Model Weights
Inside your final training run directory (`runs/classify/train-13/`), the pipeline generates the following output files for analysis and inferencing:
* **`weights/best.pt`**: The model weights that achieved the highest Top-1 validation accuracy during the final training. **Use this file for custom inference.**
* **`weights/last.pt`**: The model weights at the final epoch of training (epoch 30).
* **`results.csv` & `results.png`**: The metrics log containing loss and classification accuracy (Top-1 / Top-5) plotted over all 30 training epochs.
* **`confusion_matrix.png` & `confusion_matrix_normalized.png`**: Heatmaps showing correct classifications and any confusion across your custom classes.
* **`train_batch*.jpg`**: Visualization of the training image batch and augmentations used during the training process.
* **`val_batch0_labels.jpg` & `val_batch0_pred.jpg`**: Side-by-side visualization comparing the ground-truth classes against the model's actual predictions on the validation set.
* **`args.yaml`**: The exact configuration and hyperparameter settings used for this training run.

---

## Environment Setup

The deep learning and telemetry scripts are designed to run in a Conda environment containing PyTorch (with CUDA support), OpenCV, Ultralytics, and Optuna.

### Activate Conda Environment
To execute the scripts using the dedicated interpreter:
```bash
C:\Users\aryan\.conda\envs\training_env\python.exe <script_name>.py [args]
```

---

## Execution Guide

### Task 1: Arrow Star Pattern
```bash
python task_pattern.py -n 5
```

### Task 2: Telemetry Logger
```bash
# Logs metrics every 2 seconds for 3 iterations:
python task_logger.py --interval 2 --count 3
```

### Task 3: ASCII Reduction
```bash
python task_ascii.py --name Aryan
```

### Task 4: Hyperparameter Optimization & Model Training
Run the hyperparameter tuning and model training script using the GPU:
```bash
python task_model.py --trials 5 --epochs 30
```

---

## Submission Steps (Git Setup)

To push the clean workspace to your public repository:
1. Initialize the git repository:
   ```bash
   git init
   ```
2. Stage and commit all project files (including training runs):
   ```bash
   git add .
   git commit -m "Initial commit: Completed training tasks with Optuna integration and runs history"
   ```
3. Push to your GitHub repository:
   ```bash
   git branch -M main
   git remote add origin https://github.com/aryankumarsahaer/training-assessment-aryan.git
   git push -u origin main
   ```
