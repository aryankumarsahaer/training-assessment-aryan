import os
import cv2
import shutil
import random
import sys
import argparse
import torch
import optuna
from ultralytics import YOLO

CLASSES = ["QR", "qr", "cosgign_no", "reg_no"]
dataset_base = r"D:\Eternal Robotics\Training\1st_model\test2"
images_src = os.path.join(dataset_base, "scanned_images", "scanned_images")
labels_src = os.path.join(dataset_base, "labels", "scanned_images")

def validate_source_paths() -> list:
    if not os.path.exists(dataset_base):
        raise FileNotFoundError(f"Base dataset folder does not exist: {dataset_base}")
    if not os.path.exists(images_src) or not os.path.exists(labels_src):
        raise FileNotFoundError("Images or labels directory missing.")
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_src) if f.lower().endswith(valid_exts)]
    labels = [f for f in os.listdir(labels_src) if f.lower().endswith('.txt')]
    
    if not images or not labels:
        raise ValueError("No valid images or labels found.")
        
    return sorted(images)

def prepare_classifier_data(images: list) -> str:
    out_dir = os.path.join(dataset_base, "classifier_dataset")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    crops_by_class = {cls: [] for cls in CLASSES}

    for img_name in images:
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(labels_src, lbl_name)
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(os.path.join(images_src, img_name))
        if img is None:
            continue
        h, w, _ = img.shape

        with open(lbl_path, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                if class_id >= len(CLASSES):
                    continue
                
                x_c, y_c, bw, bh = map(float, parts[1:5])
                x1 = max(0, int((x_c - bw / 2) * w))
                y1 = max(0, int((y_c - bh / 2) * h))
                x2 = min(w, int((x_c + bw / 2) * w))
                y2 = min(h, int((y_c + bh / 2) * h))

                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    crop_name = f"{os.path.splitext(img_name)[0]}_crop_{idx}.jpg"
                    crops_by_class[CLASSES[class_id]].append({"image": crop, "name": crop_name})

    random.seed(42)
    for cls, crops in crops_by_class.items():
        if not crops:
            continue
        random.shuffle(crops)
        if len(crops) == 1:
            for split in ["train", "val"]:
                cv2.imwrite(os.path.join(out_dir, split, cls, crops[0]["name"]), crops[0]["image"])
        else:
            split_idx = max(1, int(len(crops) * 0.8))
            if split_idx == len(crops):
                split_idx = len(crops) - 1
            for item in crops[:split_idx]:
                cv2.imwrite(os.path.join(out_dir, "train", cls, item["name"]), item["image"])
            for item in crops[split_idx:]:
                cv2.imwrite(os.path.join(out_dir, "val", cls, item["name"]), item["image"])
                
    return out_dir

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna for YOLOv8 Classification.")
    parser.add_argument("-t", "--trials", type=int, default=3, help="Number of Optuna trials")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Final training epochs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA-enabled GPU is not available! GPU training is strictly required.")
        sys.exit(1)
        
    print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    
    try:
        images = validate_source_paths()
        dataset_path = prepare_classifier_data(images)
        
        def objective(trial):
            # Suggest hyperparameters
            lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
            optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
            freeze = trial.suggest_int("freeze", 0, 8)
            
            print(f"\n--- Starting Trial {trial.number}: lr0={lr0:.6f}, optimizer={optimizer}, freeze={freeze} ---")
            
            # Load fresh model weights to prevent cross-trial leakage
            model = YOLO("yolov8n-cls.pt")
            
            # Train for 1 epoch to evaluate configuration
            results = model.train(
                data=dataset_path,
                epochs=1,
                imgsz=224,
                device=0,
                lr0=lr0,
                optimizer=optimizer,
                freeze=freeze,
                verbose=False
            )
            
            # Retrieve validation accuracy
            val_acc = results.results_dict.get("metrics/accuracy_top1", 0.0)
            print(f"Trial {trial.number} Result: Top-1 Accuracy = {val_acc:.4f}")
            return val_acc

        # Run Optuna Study to maximize top-1 validation accuracy
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials)
        
        print("\n==================================================")
        print("             OPTIMIZATION COMPLETE               ")
        print("==================================================")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Top-1 Accuracy: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("==================================================\n")
        
        # Train final model using best parameters
        best_params = study.best_params
        print(f"[INFO] Starting final training with best parameters for {args.epochs} epochs...")
        
        model = YOLO("yolov8n-cls.pt")
        model.train(
            data=dataset_path,
            epochs=args.epochs,
            imgsz=224,
            device=0,
            lr0=best_params["lr0"],
            optimizer=best_params["optimizer"],
            freeze=best_params["freeze"]
        )
        print("[SUCCESS] Classification training and hyperparameter optimization complete!")
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
