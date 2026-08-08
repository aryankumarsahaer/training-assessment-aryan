# Coding & Computer Vision Assessment - Eternal Robotics Training

This repository contains clean, modular, and optimized implementations for the technical assessment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Task 1: Symmetrical Arrow Star Pattern](#task-1-symmetrical-arrow-star-pattern-task_patternpy)
3. [Task 2: Continuous Telemetry Logger](#task-2-continuous-telemetry-logger-task_loggerpy)
4. [Task 3: ASCII Single-Digit Reduction](#task-3-ascii-single-digit-reduction-task_asciipy)
5. [Task 4: Industrial YOLO Object Detection Pipeline](#task-4-industrial-yolo-object-detection-pipeline-task_modelpy)

   * [A. Objective & Class Mapping](#a-objective--class-mapping)
   * [B. Pipeline Architecture](#b-pipeline-architecture)
   * [C. Phase 1: Data Integrity & Conversion Quality Gate](#c-phase-1-data-integrity--conversion-quality-gate)
   * [D. Data Leakage Prevention & Reproducible Splitting](#d-data-leakage-prevention--reproducible-splitting)
   * [E. Physical Visual QA & Contact Sheets](#e-physical-visual-qa--contact-sheets)
   * [F. Phase 2: Mandatory GPU Training & Mixed Precision](#f-phase-2-mandatory-gpu-training--mixed-precision)
   * [G. Controlled Hyperparameter Tuning & Benchmark Table](#g-controlled-hyperparameter-tuning--benchmark-table)
   * [H. Winning Model Selection & Promotion](#h-winning-model-selection--promotion)
   * [I. Phase 3: Model Inference](#i-phase-3-model-inference)
6. [Execution & Reproduction Guide](#execution--reproduction-guide)

---

## Project Overview

The assessment consists of four distinct engineering tasks ranging from algorithmic pattern generation and telemetry monitoring to an end-to-end deep learning object detection pipeline built with PyTorch and Ultralytics YOLO.

All logic for each task is self-contained in its designated Python script without external file dependencies.

### Video Folder

[View All Task Videos on Google Drive](https://drive.google.com/drive/folders/1iy91E8ZmSG-teMduLTWoGDwSTpjLJST0?usp=sharing)

---

## Task 1: Symmetrical Arrow Star Pattern (`task_pattern.py`)

### Video Demonstration

[Watch Task 1 Video](https://drive.google.com/file/d/1XCN_tlsquhX5kP0aftEzRgQg2m72zgnD/view?usp=sharing)

### Overview & Logic

Generates a dynamic arrow-like star pattern using standard text rendering. The pattern dynamically scales for any odd integer $N \geq 5$.

### Key Algorithmic Rules:

1. **Input Validation**: Rejects even integers or values $N < 5$.
2. **Symmetry**: Computes the midpoint $\text{mid} = \lfloor N / 2 \rfloor$.
3. **Row Layout**:

   * **Upper Section ($i < \text{mid}$)**: Left stem `*`, followed by $N$ spaces, and expanding arrowhead `*` $\times (i + 1)$.
   * **Center Shaft ($i == \text{mid}$)**: Solid continuous line of stars of length $(N + \text{mid} + 1)$.
   * **Lower Section ($i > \text{mid}$)**: Left stem `*`, followed by $N$ spaces, and contracting arrowhead `*` $\times (N - i)$.

### Usage & Example:

```bash
python task_pattern.py
# Input: 5
```

**Output ($N=5$):**

```text
*     *
*     **
*********
*     **
*     *
```

---

## Task 2: Continuous Telemetry Logger (`task_logger.py`)

### Video Demonstration

[Watch Task 2 Video](https://drive.google.com/file/d/1hA0SDT5k8_XmUmIguGsOMlh_f_pXQMAt/view?usp=sharing)

### Overview & Logic

A lightweight system diagnostics utility designed for continuous background telemetry recording. It logs sequential counter steps and timestamp records directly to `telemetry.json`.

### Features:

* **Fault-Tolerant File Handling**: Automatically initializes `telemetry.json` or appends to existing records without data corruption.
* **Graceful Termination**: Captures `KeyboardInterrupt` (`Ctrl+C`) to terminate cleanly.
* **Structured JSON Schema**:

  ```json
  [
      {
          "time": "02:12:45",
          "date": "2026-08-08",
          "counter": 10
      }
  ]
  ```

### Usage:

```bash
python task_logger.py
```

---

## Task 3: ASCII Single-Digit Reduction (`task_asciipy`)

### Video Demonstration

[Watch Task 3 Video](https://drive.google.com/file/d/1kfZXwvX9Vwv6fnluZtM-zsXSRyf9suBN/view?usp=sharing)

### Overview & Logic

A recursive mathematical reduction algorithm that processes a name string into ASCII values, computes their total sum, and recursively reduces the sum to a single-digit integer (Digital Root).

### Algorithmic Breakdown:

1. **ASCII Transformation**: Converts each character to its ASCII code via `ord(char)` and computes the initial sum.
2. **Recursive Digit Reduction**:
   $$\text{reduce}(S) = \begin{cases} S & \text{if } S < 10 \ \text{reduce}(\sum d_i) & \text{where } d_i \text{ are digits of } S \end{cases}$$
3. **Full Calculation Trace**: Prints the intermediate step breakdown at each recursion level.

### Usage & Example:

```bash
python task_ascii.py
# Enter your name: ARYAN
```

**Output:**

```text
ASCII Values:
A -> 65
R -> 82
Y -> 89
A -> 65
N -> 78

ASCII Sum = 379

Digit Sum Process:
3 + 7 + 9 = 19
1 + 9 = 10
1 + 0 = 1

Final Single Digit = 1
```

---

## Task 4: Industrial YOLO Object Detection Pipeline (`task_model.py`)

### Video Demonstration

[watch Task 4 Video ](https://drive.google.com/file/d/1fJaHAPabhHl_ogqXurASJbTlEocye9Xm/view?usp=sharing)

All logic for Task 4 is encapsulated within **[task_model.py](file:///D:/Eternal%20Robotics/Training/HW_/task_model.py)** without secondary helper scripts.

---

### A. Objective & Class Mapping

* **Primary Objective**: Train an object detector for industrial verification using bounding boxes.
* **Target Classes**:

  * **`0 = barcode`** (Industrial barcode strips)
  * **`1 = hinge`** (Metal hinge brackets with center bolt)

> [!NOTE]
> In raw CVAT labels, hinges were labeled as class `0` and barcodes as class `1`. The converter automatically performs target class mapping during parsing so the resulting detector adheres to `0: barcode` and `1: hinge`.

---

### B. Pipeline Architecture

[task_model.py](file:///D:/Eternal%20Robotics/Training/HW_/task_model.py) implements three CLI execution modes:

1. `python task_model.py` (Default): Executes **Phase 1** (data validation, polygon-to-bbox conversion, reproducible splitting, visual QA overlays, and validation contact sheet generation) and stops for manual verification.
2. `python task_model.py --train`: Executes **Phase 2** (GPU-enforced baseline training, controlled hyperparameter tuning, experiment logging, and best model promotion).
3. `python task_model.py --predict <image_path>`: Executes **Phase 3** (inference on an unseen image using the promoted `best_model/best.pt`).

---

### C. Phase 1: Data Integrity & Conversion Quality Gate

Before training, 19 integrity checks are executed across all 93 source images and 91 label files:

```text
DATASET INTEGRITY REPORT
================================
Annotation format:      CVAT Normalized Polygons (x1 y1 ... xn yn)
Total Source Images:    93
Total Source Labels:    91
Barcode Objects:        89 (Target Class 0)
Hinge Objects:          56 (Target Class 1)
Missing Labels:         2 (img_033, img_035 - treated as negative background images)
Missing Images:         0
Malformed Lines:        0
Corrupt Images:         0
Invalid BBoxes:         0
```

#### Polygon-to-Bounding-Box Conversion:

For polygon coordinate sequences $(x_1, y_1, x_2, y_2, \dots, x_k, y_k)$:
$$x_{\text{min}} = \min(x_i), \quad x_{\text{max}} = \max(x_i), \quad y_{\text{min}} = \min(y_i), \quad y_{\text{max}} = \max(y_i)$$
$$x_{\text{center}} = \frac{x_{\text{min}} + x_{\text{max}}}{2}, \quad y_{\text{center}} = \frac{y_{\text{min}} + y_{\text{max}}}{2}$$
$$\text{width} = x_{\text{max}} - x_{\text{min}}, \quad \text{height} = y_{\text{max}} - y_{\text{min}}$$

---

### D. Data Leakage Prevention & Reproducible Splitting

1. **Exact Duplicate Grouping**:

   * Analysis identified 12 exact binary duplicate files (named with ` (2)` suffixes, e.g. `img_049.jpg` and `img_049 (2).jpg`).
   * Grouped by base stem prior to splitting to prevent exact copies from appearing in both train and validation sets.
2. **Sequential Similarity Analysis**:

   * Analyzed consecutive video frames using downsampled grayscale pixel Mean Squared Error (MSE).
   * Flagged frames with $\text{MSE} < 150$ (e.g. `img_078` and `img_079`, $\text{MSE} = 114.37$) and logged them in `qa/annotation_qa_report.json`.
3. **Deterministic Manifests**:

   * Split ratio: **80% Train / 20% Validation** with fixed random seed $\text{SEED} = 42$.
   * Manifests saved to `dataset/train.txt` (75 images) and `dataset/val.txt` (18 images).

```text
SPLIT BREAKDOWN
--------------------------------
Train Set (64 unique stems / 75 images):
  - Barcode: 74
  - Hinge:   44

Validation Set (17 unique stems / 18 images):
  - Barcode: 15
  - Hinge:   12
```

---

### E. Physical Visual QA & Contact Sheets

To verify bounding box alignment before GPU training:

* **Visual Overlays**: Generated annotated preview copies with color-coded bounding boxes and label tags (`barcode [0]` in Cyan, `hinge [1]` in Orange) under `qa/val/`.
* **Contact Sheets**: Assembled validation previews into structured grid sheets:

  * `qa/validation_contact_sheet_01.jpg` (Images 1–16)
  * `qa/validation_contact_sheet_02.jpg` (Images 17–18)
* **Machine-Readable Report**: Generated `qa/annotation_qa_report.json` containing complete dataset statistics and leakage analysis.

---

### F. Phase 2: Mandatory GPU Training & Mixed Precision

Training enforces GPU availability via PyTorch CUDA. If GPU is unavailable, the pipeline terminates immediately.

```text
TRAINING ENVIRONMENT
================================
Conda Environment:   training_env
PyTorch Version:     2.5.1+cu121
CUDA Version:        12.1
Ultralytics Version: 8.4.115
GPU Device:          NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM)
Mixed Precision:     AMP Enabled (Automatic Mixed Precision)
Max Epoch Budget:    200 epochs
Early Stopping:      Patience = 30 epochs
```

---

### G. Controlled Hyperparameter Tuning & Benchmark Table

Rather than unguided brute-force search, five structured experiments were executed sequentially to isolate the effects of resolution, model capacity, optimizer choice, and augmentation:

| Exp ID                  | Model Architecture | Resolution (`imgsz`) | Batch Size | Optimizer      | Learning Rate (`lr0`) | Weight Decay | Epochs Completed | Best Epoch | Precision  | Recall     | mAP@50     | mAP@50-95  | Barcode AP@50-95 | Hinge AP@50-95 |
| :---------------------- | :----------------- | :------------------- | :--------- | :------------- | :-------------------- | :----------- | :--------------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------------- | :------------- |
| **exp_001** (Baseline)  | `yolov8n.pt`       | 640                  | 16         | auto (SGD)     | 0.010                 | 0.0005       | 137              | 107        | 0.9908     | 0.9583     | 0.9759     | 0.7735     | 0.7259           | 0.8212         |
| **exp_002** (Winning)   | **`yolov8n.pt`**   | **800**              | **8**      | **auto (SGD)** | **0.010**             | **0.0005**   | **122**          | **92**     | **0.9708** | **0.9969** | **0.9950** | **0.7790** | **0.7474**       | **0.8107**     |
| **exp_003** (Capacity)  | `yolov8s.pt`       | 800                  | 8          | auto (SGD)     | 0.010                 | 0.0005       | 122              | 92         | 0.9813     | 0.9550     | 0.9550     | 0.7122     | 0.7256           | 0.6987         |
| **exp_004** (AdamW)     | `yolov8s.pt`       | 800                  | 8          | AdamW          | 0.001                 | 0.0010       | 81               | 51         | 0.9901     | 0.9583     | 0.9870     | 0.7642     | 0.7141           | 0.8142         |
| **exp_005** (No Mosaic) | `yolov8s.pt`       | 800                  | 8          | AdamW          | 0.001                 | 0.0010       | 87               | 57         | 0.9899     | 0.9571     | 0.9691     | 0.7541     | 0.7031           | 0.8051         |

*All training runs and experiment records are preserved in `runs/experiment_summary.csv`.*

---

### H. Winning Model Selection & Promotion

#### Selection Criterion:

Primary ranking metric is validation **`mAP@50-95`** along with class-level AP balance and generalization:

* **`exp_002`** achieved the highest validation performance (**`0.7790` mAP@50-95**, **`0.9950` mAP@50**, and **`0.9969` Recall**).
* Raising resolution to $800\text{px}$ improved small barcode feature localization ($\text{Barcode AP} = 0.7474$).
* Larger model capacity (`yolov8s` in `exp_003`) showed slight overfitting on this small custom dataset, confirming `yolov8n` as the better generalizing architecture.

#### Promoted Artifacts (`best_model/`):

```text
best_model/
├── best.pt              # Selected model weights from exp_002
├── best_config.yaml     # Complete winning hyperparameters and benchmark metrics
├── metrics.json         # Precision, recall, mAP50, mAP50-95, per-class AP
├── dataset_info.json    # Dataset partition & leakage audit
└── environment.txt      # PyTorch, CUDA, GPU, and library version details
```

---

### I. Phase 3: Model Inference

Inference can be run on any unseen image using the promoted model:

```bash
python task_model.py --predict "images/img_002.jpg"
```

**Output**:

```text
[INFO] Running inference on image: images/img_002.jpg
image 1/1: 480x800 1 barcode, 1 hinge, 9.9ms
[INFO] Saved annotated prediction image to: pred_img_002.jpg
[INFO] Detections:
  1. Class: hinge [1], Confidence: 0.9244, BBox: [434.7, 331.3, 664.5, 576.0]
  2. Class: barcode [0], Confidence: 0.8737, BBox: [291.0, 164.3, 512.2, 277.9]
```

---

## Execution & Reproduction Guide

### Environment Activation

```bash
conda activate training_env
```

### 1. Run Task 1 (Star Arrow Pattern)

```bash
python task_pattern.py
```

### 2. Run Task 2 (Telemetry Logger)

```bash
python task_logger.py
```

### 3. Run Task 3 (ASCII Reduction)

```bash
python task_ascii.py
```

### 4. Run Task 4 Pre-Training Pipeline (Validation & QA)

```bash
python task_model.py
```

### 5. Run Task 4 GPU Training & Model Selection

```bash
python task_model.py --train
```

### 6. Run Task 4 Inference

```bash
python task_model.py --predict "images/img_002.jpg"
```

---

## Git Submission Steps

```bash
git add .
git commit -m "Complete training assessment: Task 1 pattern, Task 2 telemetry, Task 3 ASCII, and Task 4 YOLO object detection pipeline"
git push origin main
```
