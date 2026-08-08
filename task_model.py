import os
import sys
import time
import json
import random
import re
import shutil
import argparse
import csv
import logging
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch

# Setup logger to stream to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("YOLO_Pipeline")

# Constants
SEED = 42
CLASSES = {0: "barcode", 1: "hinge"}

def set_seed(seed: int = SEED):
    """
    Sets seeds for reproducibility across random, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def get_base_stem(filename: str) -> str:
    """
    Extracts the base filename stem by removing file extensions and exact
    duplicate suffix markers (like ' (2)').
    """
    stem = os.path.splitext(filename)[0]
    base = re.sub(r'\s*\(\d+\)\s*$', '', stem)
    return base.strip()

def polygon_to_bbox(parts: list, line_num: int, filename: str) -> tuple:
    """
    Parses CVAT polygon coordinate lines and converts them to standard axis-aligned
    YOLO bounding box coordinate lines. If the line is already a YOLO bounding box,
    validates the coordinate format and range.
    
    Returns: (class_id, x_center, y_center, width, height, is_polygon)
    """
    if not parts:
        raise ValueError("Empty annotation line")
        
    try:
        raw_class_id = int(parts[0])
    except ValueError:
        raise ValueError(f"Line {line_num}: Class ID must be an integer, got '{parts[0]}'")
        
    if raw_class_id not in [0, 1]:
        raise ValueError(f"Line {line_num}: Invalid raw class ID {raw_class_id}. Allowed classes: [0, 1]")
        
    # Map raw class ID (0 = hinge, 1 = barcode) to target class ID (0 = barcode, 1 = hinge)
    class_id = 1 if raw_class_id == 0 else 0
        
    # Standard YOLO detection format: class_id + 4 normalized coords = 5 elements
    if len(parts) == 5:
        try:
            x_center, y_center, width, height = map(float, parts[1:5])
        except ValueError:
            raise ValueError(f"Line {line_num}: Coordinates must be floating point numbers")
            
        for name, val in [("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)]:
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Line {line_num}: {name} value {val} is outside [0, 1] range")
                
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Line {line_num}: Bounding box width/height must be positive")
            
        return class_id, x_center, y_center, width, height, False
        
    # Polygon format: class_id + even number of coordinates (min 6 values for 3 points)
    coords_str = parts[1:]
    if len(coords_str) % 2 != 0:
        raise ValueError(f"Line {line_num}: Odd number of coordinate values ({len(coords_str)})")
        
    if len(coords_str) < 6:
        raise ValueError(f"Line {line_num}: Insufficient coordinates for a polygon ({len(coords_str)} values, min 6)")
        
    try:
        coords = [float(c) for c in coords_str]
    except ValueError:
        raise ValueError(f"Line {line_num}: Polygon coordinates must be numeric")
        
    xs = coords[0::2]
    ys = coords[1::2]
    
    # Validate boundary coordinate checks
    for x in xs:
        if x < 0.0 or x > 1.0:
            raise ValueError(f"Line {line_num}: Polygon X coordinate {x} is outside [0, 1] range")
    for y in ys:
        if y < 0.0 or y > 1.0:
            raise ValueError(f"Line {line_num}: Polygon Y coordinate {y} is outside [0, 1] range")
            
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Line {line_num}: Bounding box has zero/negative area (width={width}, height={height})")
        
    return class_id, x_center, y_center, width, height, True

def validate_raw_dataset(image_dir: Path, label_dir: Path) -> dict:
    """
    Performs 19 pre-training validation checks on the raw dataset.
    """
    logger.info("Executing Phase 1 pre-training raw dataset validation checks...")
    
    valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in valid_img_exts])
    lbl_files = sorted([f for f in label_dir.iterdir() if f.suffix.lower() == ".txt"])
    
    img_stems = {f.stem for f in img_files}
    lbl_stems = {f.stem for f in lbl_files}
    
    missing_labels = sorted(list(img_stems - lbl_stems))
    missing_images = sorted(list(lbl_stems - img_stems))
    
    corrupt_images = []
    empty_labels = []
    malformed_annotations = []
    invalid_coords = []
    invalid_classes = []
    invalid_boxes = []
    
    barcode_count = 0
    hinge_count = 0
    total_polygons = 0
    total_bboxes = 0
    
    # Corrupt/unreadable images checks
    for img_path in img_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt_images.append(img_path.name)
        except Exception:
            corrupt_images.append(img_path.name)
            
    # Label validations
    for lbl_path in lbl_files:
        if lbl_path.stat().st_size == 0:
            empty_labels.append(lbl_path.name)
            continue
            
        try:
            with open(lbl_path, "r") as f:
                lines = f.read().splitlines()
        except Exception as e:
            malformed_annotations.append(f"{lbl_path.name} (IO Error: {e})")
            continue
            
        has_annotations = False
        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
                
            try:
                class_id, _, _, _, _, is_poly = polygon_to_bbox(parts, idx, lbl_path.name)
                has_annotations = True
                
                if class_id == 0:
                    barcode_count += 1
                elif class_id == 1:
                    hinge_count += 1
                    
                if is_poly:
                    total_polygons += 1
                else:
                    total_bboxes += 1
                    
            except Exception as e:
                err_msg = str(e)
                if "Class ID" in err_msg or "Invalid class ID" in err_msg:
                    invalid_classes.append(f"{lbl_path.name}:L{idx} - {err_msg}")
                elif "outside [0, 1]" in err_msg:
                    invalid_coords.append(f"{lbl_path.name}:L{idx} - {err_msg}")
                elif "positive" in err_msg or "zero/negative" in err_msg:
                    invalid_boxes.append(f"{lbl_path.name}:L{idx} - {err_msg}")
                else:
                    malformed_annotations.append(f"{lbl_path.name}:L{idx} - {err_msg}")
                    
        if not has_annotations:
            empty_labels.append(lbl_path.name)
            
    return {
        "total_images": len(img_files),
        "total_labels": len(lbl_files),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "empty_labels": empty_labels,
        "malformed_annotations": malformed_annotations,
        "invalid_coords": invalid_coords,
        "invalid_classes": invalid_classes,
        "invalid_boxes": invalid_boxes,
        "barcode_count": barcode_count,
        "hinge_count": hinge_count,
        "total_polygon_annotations": total_polygons,
        "total_bbox_annotations": total_bboxes,
        "img_files": img_files,
        "lbl_files": lbl_files
    }

def analyze_sequential_leakage(image_dir: Path, unique_stems: list, img_files: list) -> list:
    """
    Performs perceptual/visual similarity check on adjacent sequential frames.
    """
    logger.info("Inspecting sequential frame similarity to assess validation leakage risk...")
    sorted_stems = sorted(unique_stems)
    leakage_risks = []
    
    stem_to_file = {}
    for f in img_files:
        base_stem = get_base_stem(f.name)
        if base_stem not in stem_to_file:
            stem_to_file[base_stem] = f
            
    for i in range(len(sorted_stems) - 1):
        stem1, stem2 = sorted_stems[i], sorted_stems[i+1]
        
        # Check if stems end in sequential numbers
        digits1 = re.findall(r'\d+', stem1)
        digits2 = re.findall(r'\d+', stem2)
        if not digits1 or not digits2:
            continue
            
        num1 = int(digits1[-1])
        num2 = int(digits2[-1])
        if abs(num1 - num2) != 1:
            continue
            
        f1 = stem_to_file.get(stem1)
        f2 = stem_to_file.get(stem2)
        if not f1 or not f2:
            continue
            
        try:
            img1 = cv2.imread(str(f1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(f2), cv2.IMREAD_GRAYSCALE)
            if img1 is not None and img2 is not None and img1.shape == img2.shape:
                # Downsample for faster and robust comparison
                img1_small = cv2.resize(img1, (128, 128))
                img2_small = cv2.resize(img2, (128, 128))
                mse = float(np.mean((img1_small.astype(np.float32) - img2_small.astype(np.float32)) ** 2))
                
                # Report if similarity is high
                if mse < 150.0:
                    leakage_risks.append({
                        "frame1": stem1,
                        "frame2": stem2,
                        "mse": round(mse, 2),
                        "description": f"High sequential similarity detected between {stem1} and {stem2} (MSE: {mse:.2f})"
                    })
        except Exception as e:
            logger.warning(f"Error comparing frames {stem1} and {stem2}: {e}")
            
    return leakage_risks

def create_train_val_split(img_files: list, seed: int = SEED, split_ratio: float = 0.8) -> tuple:
    """
    Groups image filenames by base stem and splits them reproducibly (80/20).
    This ensures identical duplicate files are mapped to the same split.
    """
    duplicate_groups = {}
    for f in img_files:
        base_stem = get_base_stem(f.name)
        duplicate_groups.setdefault(base_stem, []).append(f.name)
        
    unique_base_stems = sorted(list(duplicate_groups.keys()))
    
    # Shuffle uniquely grouped stems with fixed seed
    rng = random.Random(seed)
    rng.shuffle(unique_base_stems)
    
    split_idx = int(len(unique_base_stems) * split_ratio)
    train_stems = set(unique_base_stems[:split_idx])
    val_stems = set(unique_base_stems[split_idx:])
    
    return train_stems, val_stems, duplicate_groups

def prepare_dataset(
    image_dir: Path,
    label_dir: Path,
    dataset_dir: Path,
    train_stems: set,
    val_stems: set,
    duplicate_groups: dict
) -> tuple:
    """
    Builds the dataset/ directory structure. Copies images, writes converted
    labels, and generates train.txt / val.txt manifests.
    """
    logger.info("Generating YOLO prepared dataset splits...")
    
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    train_manifest = []
    val_manifest = []
    
    barcode_counts = {"train": 0, "val": 0}
    hinge_counts = {"train": 0, "val": 0}
    
    for base_stem, filenames in duplicate_groups.items():
        split = "train" if base_stem in train_stems else "val"
        manifest = train_manifest if split == "train" else val_manifest
        
        for img_name in filenames:
            stem = os.path.splitext(img_name)[0]
            src_img_path = image_dir / img_name
            dst_img_path = dataset_dir / "images" / split / img_name
            
            # Copy image file
            shutil.copy2(src_img_path, dst_img_path)
            # Manifest entry relative path
            manifest.append(f"./dataset/images/{split}/{img_name}")
            
            # Convert label file
            src_lbl_name = f"{stem}.txt"
            src_lbl_path = label_dir / src_lbl_name
            dst_lbl_path = dataset_dir / "labels" / split / f"{stem}.txt"
            
            converted_lines = []
            if src_lbl_path.exists():
                try:
                    with open(src_lbl_path, "r") as f:
                        lines = f.read().splitlines()
                    for idx, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if not parts:
                            continue
                        class_id, x_c, y_c, w, h, _ = polygon_to_bbox(parts, idx, src_lbl_name)
                        converted_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                        
                        if class_id == 0:
                            barcode_counts[split] += 1
                        elif class_id == 1:
                            hinge_counts[split] += 1
                except Exception as e:
                    logger.warning(f"Failed to convert label {src_lbl_name}: {e}. Writing empty label.")
                    converted_lines = []
            
            # Write label file (empty if background)
            with open(dst_lbl_path, "w") as f:
                f.write("\n".join(converted_lines) + ("\n" if converted_lines else ""))
                
    # Write train.txt and val.txt
    with open(dataset_dir / "train.txt", "w") as f:
        f.write("\n".join(sorted(train_manifest)) + "\n")
    with open(dataset_dir / "val.txt", "w") as f:
        f.write("\n".join(sorted(val_manifest)) + "\n")
        
    return barcode_counts, hinge_counts

def generate_qa_images(dataset_dir: Path, qa_dir: Path):
    """
    Creates validation image copies with bounding box overlays from converted YOLO labels.
    """
    logger.info("Generating validation & training QA overlay images...")
    
    # BGR format colors
    colors = {
        0: (255, 255, 0),    # Cyan for barcode
        1: (0, 165, 255)     # Orange for hinge
    }
    
    for split in ["train", "val"]:
        split_img_dir = dataset_dir / "images" / split
        split_lbl_dir = dataset_dir / "labels" / split
        out_qa_dir = qa_dir / split
        out_qa_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_img_dir.iterdir():
            if not img_path.is_file():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w, _ = img.shape
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    
                    xmin = int((x_c - bw / 2.0) * w)
                    ymin = int((y_c - bh / 2.0) * h)
                    xmax = int((x_c + bw / 2.0) * w)
                    ymax = int((y_c + bh / 2.0) * h)
                    
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(w - 1, xmax), min(h - 1, ymax)
                    
                    color = colors.get(class_id, (0, 255, 0))
                    class_name = CLASSES.get(class_id, "unknown")
                    
                    # Draw box
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    # Draw label background
                    label_text = f"{class_name} [{class_id}]"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    lbl_ymin = max(text_h + 5, ymin)
                    cv2.rectangle(img, (xmin, lbl_ymin - text_h - 4), (xmin + text_w + 6, lbl_ymin + baseline - 2), color, -1)
                    
                    # Draw text
                    cv2.putText(img, label_text, (xmin + 3, lbl_ymin - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                    
            cv2.imwrite(str(out_qa_dir / img_path.name), img)

def generate_validation_contact_sheet(qa_dir: Path, output_path: Path):
    """
    Arranges annotated validation images in a readable grid.
    """
    logger.info("Assembling validation contact sheet grid...")
    val_qa_dir = qa_dir / "val"
    if not val_qa_dir.exists():
        logger.warning("Validation QA dir missing. Skipping contact sheet.")
        return
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    qa_images = sorted([f for f in val_qa_dir.iterdir() if f.suffix.lower() in valid_exts])
    
    if not qa_images:
        logger.warning("No validation QA images found. Skipping contact sheet.")
        return
        
    # Standard grid cell size
    cell_w, cell_h = 512, 288
    cols = 4
    images_per_sheet = 16
    sheets_count = (len(qa_images) + images_per_sheet - 1) // images_per_sheet
    
    for sheet_idx in range(sheets_count):
        sheet_imgs = qa_images[sheet_idx * images_per_sheet : (sheet_idx + 1) * images_per_sheet]
        rows = (len(sheet_imgs) + cols - 1) // cols
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for idx, img_path in enumerate(sheet_imgs):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_resized = cv2.resize(img, (cell_w, cell_h))
            
            # File tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            text = img_path.name
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img_resized, (5, 5), (5 + text_w + 6, 5 + text_h + baseline + 2), (0, 0, 0), -1)
            cv2.putText(img_resized, text, (8, 5 + text_h + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            r = idx // cols
            c = idx % cols
            canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = img_resized
            
        if sheets_count == 1:
            dest_path = output_path
        else:
            dest_path = output_path.parent / f"{output_path.stem}_{sheet_idx + 1:02d}{output_path.suffix}"
            
        cv2.imwrite(str(dest_path), canvas)
        logger.info(f"Validation contact sheet created successfully: {dest_path}")

def save_qa_report(
    report_path: Path,
    raw_stats: dict,
    train_stems: set,
    val_stems: set,
    duplicate_groups: dict,
    leakage_risks: list
):
    """
    Writes machine-readable dataset statistics to qa/annotation_qa_report.json.
    """
    report_data = {
        "report_generated_at": datetime.now().isoformat(),
        "split_seed": SEED,
        "split_ratio_train_val": "80/20",
        "counts": {
            "source_images": raw_stats["total_images"],
            "source_labels": raw_stats["total_labels"],
            "train_images": len(train_stems),
            "val_images": len(val_stems),
            "total_barcode_instances": raw_stats["barcode_count"],
            "total_hinge_instances": raw_stats["hinge_count"]
        },
        "annotations_format": {
            "polygons": raw_stats["total_polygon_annotations"],
            "bounding_boxes": raw_stats["total_bbox_annotations"]
        },
        "validation_issues": {
            "missing_labels": raw_stats["missing_labels"],
            "missing_images": raw_stats["missing_images"],
            "corrupt_images": raw_stats["corrupt_images"],
            "empty_labels": raw_stats["empty_labels"],
            "malformed_annotations": raw_stats["malformed_annotations"],
            "invalid_coords": raw_stats["invalid_coords"],
            "invalid_classes": raw_stats["invalid_classes"],
            "invalid_boxes": raw_stats["invalid_boxes"]
        },
        "duplicate_groups": {k: v for k, v in duplicate_groups.items() if len(v) > 1},
        "leakage_analysis": {
            "sequential_leakage_risks": leakage_risks
        }
    }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"Annotation QA Report saved to {report_path}")

def run_pretraining_phase(project_dir: Path) -> dict:
    """
    Executes Phase 1 data pipeline validation, conversion, split, and QA.
    """
    image_dir = project_dir / "images"
    label_dir = project_dir / "labels"
    dataset_dir = project_dir / "dataset"
    qa_dir = project_dir / "qa"
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Source images directory missing: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Source labels directory missing: {label_dir}")
        
    set_seed(SEED)
    
    # 1. Dataset validation
    raw_stats = validate_raw_dataset(image_dir, label_dir)
    
    # 2. Extract base stems and split
    train_stems, val_stems, duplicate_groups = create_train_val_split(raw_stats["img_files"])
    
    # 3. Leakage Analysis
    leakage_risks = analyze_sequential_leakage(image_dir, list(duplicate_groups.keys()), raw_stats["img_files"])
    
    # 4. Prepare dataset and write manifests
    barcode_counts, hinge_counts = prepare_dataset(
        image_dir, label_dir, dataset_dir, train_stems, val_stems, duplicate_groups
    )
    
    # 5. Visual QA Overlays
    generate_qa_images(dataset_dir, qa_dir)
    
    # 6. Grid Contact Sheet
    contact_sheet_path = qa_dir / "validation_contact_sheet.jpg"
    generate_validation_contact_sheet(qa_dir, contact_sheet_path)
    
    # 7. Machine-readable report
    qa_report_path = qa_dir / "annotation_qa_report.json"
    save_qa_report(qa_report_path, raw_stats, train_stems, val_stems, duplicate_groups, leakage_risks)
    
    # Print Pre-training dataset report
    print("\nDATASET REPORT")
    print("================================")
    print(f"Annotation format: {'Polygon' if raw_stats['total_polygon_annotations'] > 0 else 'Bounding Box'}")
    print(f"Source images: {raw_stats['total_images']}")
    print(f"Source labels: {raw_stats['total_labels']}")
    print(f"Barcode objects: {raw_stats['barcode_count']}")
    print(f"Hinge objects: {raw_stats['hinge_count']}")
    print(f"Missing labels: {len(raw_stats['missing_labels'])}")
    print(f"Missing images: {len(raw_stats['missing_images'])}")
    print(f"Malformed annotations: {len(raw_stats['malformed_annotations'])}")
    print(f"Corrupt images: {len(raw_stats['corrupt_images'])}")
    print(f"Invalid boxes: {len(raw_stats['invalid_boxes'])}")
    print("\nSPLIT")
    print("--------------------------------")
    print(f"Seed: {SEED}")
    print(f"Train images: {len(train_stems)}")
    print(f"Validation images: {len(val_stems)}")
    print("\nTrain:")
    print(f"Barcode: {barcode_counts['train']}")
    print(f"Hinge: {hinge_counts['train']}")
    print("\nValidation:")
    print(f"Barcode: {barcode_counts['val']}")
    print(f"Hinge: {hinge_counts['val']}")
    print("\nQA")
    print("--------------------------------")
    print(f"Train previews:\n  {qa_dir}/train/")
    print(f"Validation previews:\n  {qa_dir}/val/")
    print(f"Contact sheet:\n  {contact_sheet_path}")
    print(f"Annotation QA Report:\n  {qa_report_path}")
    print("\nSTATUS:\nREADY FOR MANUAL VERIFICATION")
    
    return raw_stats

# Phase 2 - GPU Training Phase
def create_data_yaml(project_dir: Path, dataset_dir: Path) -> Path:
    """
    Creates data.yaml for YOLO training referencing absolute directories.
    """
    data_yaml_path = project_dir / "data.yaml"
    yaml_content = f"""path: {dataset_dir.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: barcode
  1: hinge
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"Created data.yaml config: {data_yaml_path}")
    return data_yaml_path

def train_experiments(project_dir: Path, data_yaml_path: Path):
    """
    GPU-enforced controlled training experiments.
    """
    if not torch.cuda.is_available():
        logger.error("CUDA-enabled GPU is NOT available! GPU training is strictly required.")
        sys.exit("Error: CUDA GPU training is mandatory for this project. Check your training_env setup.")
        
    device = 0
    import ultralytics
    from ultralytics import YOLO
    
    logger.info("\nTRAINING ENVIRONMENT\n"
                "================================\n"
                f"Conda environment: training_env\n"
                f"PyTorch: {torch.__version__}\n"
                f"Ultralytics: {ultralytics.__version__}\n"
                f"CUDA available: True\n"
                f"CUDA version: {torch.version.cuda}\n"
                f"GPU: {torch.cuda.get_device_name(device)}\n"
                f"Device: cuda:{device}\n"
                "================================\n")
                
    summary_path = project_dir / "runs" / "experiment_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    csv_header = [
        "experiment_id", "model_type", "imgsz", "batch", "optimizer", 
        "lr0", "weight_decay", "epochs_requested", "epochs_completed", 
        "best_epoch", "early_stopped", "precision", "recall", 
        "mAP50", "mAP50_95", "barcode_ap", "hinge_ap", "best_weight_path"
    ]
    
    if not summary_path.exists():
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            
    # Staged experiment configs
    configs = [
        # Baseline yolov8n
        {"id": "exp_001", "model": "yolov8n.pt", "imgsz": 640, "optimizer": "auto", "lr0": 0.01, "weight_decay": 0.0005, "augment": True},
        # Baseline + imgsz=800
        {"id": "exp_002", "model": "yolov8n.pt", "imgsz": 800, "optimizer": "auto", "lr0": 0.01, "weight_decay": 0.0005, "augment": True},
        # Larger capacity yolov8s
        {"id": "exp_003", "model": "yolov8s.pt", "imgsz": 800, "optimizer": "auto", "lr0": 0.01, "weight_decay": 0.0005, "augment": True},
        # AdamW Optimizer tuning
        {"id": "exp_004", "model": "yolov8s.pt", "imgsz": 800, "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.001, "augment": True},
        # Modest augmentations (mosaic disabled)
        {"id": "exp_005", "model": "yolov8s.pt", "imgsz": 800, "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.001, "augment": False}
    ]
    
    results_list = []
    
    for cfg in configs:
        exp_id = cfg["id"]
        logger.info(f"\n========================================\n"
                    f"STARTING EXPERIMENT: {exp_id}\n"
                    f"========================================\n"
                    f"Model: {cfg['model']}, Imgsz: {cfg['imgsz']}, Optimizer: {cfg['optimizer']}\n")
                    
        # Dynamically allocate batch size to avoid OOM on 4GB VRAM
        batch_size = 16 if cfg["imgsz"] <= 640 and "yolov8n" in cfg["model"] else 8
        
        try:
            best_weight = project_dir / "runs" / exp_id / "weights" / "best.pt"
            mosaic_val = 0.0 if not cfg["augment"] else 1.0
            
            # Check if experiment was already trained
            if best_weight.exists():
                logger.info(f"Existing checkpoint found for {exp_id}. Loading {best_weight} for evaluation...")
                model = YOLO(str(best_weight))
            else:
                model = YOLO(cfg["model"])
                results = model.train(
                    data=str(data_yaml_path.resolve()),
                    epochs=200,
                    patience=30,
                    imgsz=cfg["imgsz"],
                    batch=batch_size,
                    optimizer=cfg["optimizer"],
                    lr0=cfg["lr0"],
                    weight_decay=cfg["weight_decay"],
                    device=device,
                    seed=SEED,
                    amp=True,
                    name=exp_id,
                    project=str(project_dir / "runs"),
                    mosaic=mosaic_val,
                    verbose=True
                )
            
            # Evaluate results
            metrics = model.val(data=str(data_yaml_path.resolve()), device=device)
            results_dict = metrics.results_dict
            
            precision = float(results_dict.get("metrics/precision(B)", 0.0))
            recall = float(results_dict.get("metrics/recall(B)", 0.0))
            map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
            
            # Per-class AP metrics
            ap_50_95 = metrics.box.ap
            barcode_ap = float(ap_50_95[0]) if len(ap_50_95) > 0 else 0.0
            hinge_ap = float(ap_50_95[1]) if len(ap_50_95) > 1 else 0.0
            
            # Safely extract training progress metrics from results.csv
            results_csv_path = project_dir / "runs" / exp_id / "results.csv"
            epochs_completed = 0
            best_epoch = 0
            early_stopped = False
            
            if results_csv_path.exists():
                with open(results_csv_path, "r") as f:
                    lines = f.read().splitlines()
                if len(lines) > 1:
                    header = [h.strip() for h in lines[0].split(",")]
                    data_lines = [l.split(",") for l in lines[1:] if l.strip()]
                    epochs_completed = len(data_lines)
                    
                    if "metrics/mAP50-95(B)" in header:
                        map_idx = header.index("metrics/mAP50-95(B)")
                        map_vals = [float(row[map_idx]) for row in data_lines]
                        best_epoch = map_vals.index(max(map_vals)) + 1
                    else:
                        best_epoch = epochs_completed
                    early_stopped = (epochs_completed < 200) and (epochs_completed - best_epoch >= 29)
            
            row = [
                exp_id, cfg["model"], cfg["imgsz"], batch_size, cfg["optimizer"],
                cfg["lr0"], cfg["weight_decay"], 200, epochs_completed,
                best_epoch, early_stopped, precision, recall, map50, map50_95,
                barcode_ap, hinge_ap, str(best_weight.resolve())
            ]
            
            with open(summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            logger.info(f"Experiment {exp_id} complete. mAP@50-95: {map50_95:.4f}")
            results_list.append(row)
            
        except Exception as e:
            logger.error(f"Experiment {exp_id} processing failed: {e}")
            torch.cuda.empty_cache()
            
    if not results_list:
        raise RuntimeError("No experiments completed successfully.")
        
    # Promote best model according to mAP50-95 and class sanity check
    sorted_results = sorted(results_list, key=lambda x: x[14], reverse=True)
    best_row = None
    for row in sorted_results:
        # Sanity check: ensure neither class is completely failing
        if row[15] > 0.01 and row[16] > 0.01:
            best_row = row
            break
            
    if best_row is None:
        logger.warning("No configuration passed class AP sanity checks. Promoting best by mAP@50-95 overall.")
        best_row = sorted_results[0]
        
    promote_best_model(project_dir, best_row)

def promote_best_model(project_dir: Path, best_row: list):
    """
    Saves promoted weights, config YAML, metrics JSON, dataset info, and env.
    """
    logger.info(f"Promoting best model checkpoint: {best_row[0]}")
    best_dir = project_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    
    src_weight = Path(best_row[17])
    dst_weight = best_dir / "best.pt"
    
    if src_weight.exists():
        shutil.copy2(src_weight, dst_weight)
    else:
        raise FileNotFoundError(f"Best weight file not found: {src_weight}")
        
    # Read prepared report info
    qa_report_path = project_dir / "qa" / "annotation_qa_report.json"
    dataset_info = {}
    if qa_report_path.exists():
        with open(qa_report_path, "r") as f:
            dataset_info = json.load(f)
            
    best_config_yaml = f"""winning_experiment_id: {best_row[0]}
model:
  architecture: {best_row[1]}
  pretrained_weights: {best_row[1]}
dataset:
  data_yaml: data.yaml
  seed: {SEED}
  split_ratio: 80/20
training:
  epochs_requested: {best_row[7]}
  epochs_completed: {best_row[8]}
  best_epoch: {best_row[9]}
  imgsz: {best_row[2]}
  batch: {best_row[3]}
  optimizer: {best_row[4]}
  lr0: {best_row[5]}
  weight_decay: {best_row[6]}
  early_stopped: {best_row[10]}
results:
  precision: {best_row[11]:.6f}
  recall: {best_row[12]:.6f}
  map50: {best_row[13]:.6f}
  map50_95: {best_row[14]:.6f}
per_class:
  barcode:
    mAP50_95: {best_row[15]:.6f}
  hinge:
    mAP50_95: {best_row[16]:.6f}
"""
    with open(best_dir / "best_config.yaml", "w") as f:
        f.write(best_config_yaml)
        
    metrics_data = {
        "precision": best_row[11],
        "recall": best_row[12],
        "mAP50": best_row[13],
        "mAP50_95": best_row[14],
        "per_class": {
            "barcode": best_row[15],
            "hinge": best_row[16]
        }
    }
    with open(best_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    with open(best_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=4)
        
    import ultralytics
    env_content = f"""Python version: {sys.version}
PyTorch version: {torch.__version__}
Ultralytics version: {ultralytics.__version__}
CUDA availability: True
CUDA version: {torch.version.cuda}
GPU name: {torch.cuda.get_device_name(0)}
device used: cuda:0
"""
    with open(best_dir / "environment.txt", "w") as f:
        f.write(env_content)
        
    logger.info(f"Promoted best model artifacts successfully to: {best_dir}")

def run_inference(image_path_str: str, project_dir: Path):
    """
    Performs inference on a custom image using the promoted best model.
    """
    best_weight_path = project_dir / "best_model" / "best.pt"
    if not best_weight_path.exists():
        logger.error(f"Promoted weight file missing: {best_weight_path}. Complete training first.")
        sys.exit(1)
        
    from ultralytics import YOLO
    model = YOLO(str(best_weight_path))
    
    logger.info(f"Running inference on image: {image_path_str}")
    results = model.predict(source=image_path_str, conf=0.25)
    
    for r in results:
        img_annotated = r.plot()
        out_name = f"pred_{os.path.basename(image_path_str)}"
        cv2.imwrite(out_name, img_annotated)
        logger.info(f"Saved annotated prediction image to: {out_name}")
        
        logger.info("Detections:")
        for idx, box in enumerate(r.boxes, 1):
            cls_id = int(box.cls[0])
            class_name = CLASSES.get(cls_id, "unknown")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            logger.info(f"  {idx}. Class: {class_name} [{cls_id}], Confidence: {conf:.4f}, BBox: {[round(x, 1) for x in xyxy]}")

def main():
    parser = argparse.ArgumentParser(description="Task 4 Object Detection Pipeline: Barcode and Hinge detector.")
    parser.add_argument("--train", action="store_true", help="Runs Phase 2 GPU training and hyperparameter optimization.")
    parser.add_argument("--predict", type=str, help="Runs inference on a custom image using the best promoted model.")
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.resolve()
    
    if args.predict:
        run_inference(args.predict, project_dir)
    elif args.train:
        dataset_dir = project_dir / "dataset"
        data_yaml_path = create_data_yaml(project_dir, dataset_dir)
        train_experiments(project_dir, data_yaml_path)
    else:
        # Default behavior: Pre-training data pipeline only
        run_pretraining_phase(project_dir)

if __name__ == "__main__":
    main()
