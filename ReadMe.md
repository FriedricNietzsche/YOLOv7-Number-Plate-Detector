# Number Plate Detector (YOLOv7) — Custom Dataset
A practical number‑plate detection project built on top of the official implementation of paper [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696). It includes a trained checkpoint (`best.pt`) on a custom dataset and ready‑to‑run scripts for image and video inference.

> Attribution: This project builds on the official YOLOv7 repository by WongKinYiu. See `LICENSE.md` for the original license and project details.

## Results (quick demo)
- Weights: `best.pt`, img-size: 640, conf-thres: 0.25, device: CPU
- Image: 2 plates detected — `runs/detect/quick2/1.jpg`
  - Preview:
    <br>
    <img src="runs/detect/quick2/1.jpg" alt="Number plate detection result on 1.jpg" width="520"/>
- Video: `runs/detect/video_quick/1.mp4` (749 frames, ~64.1 s on CPU)
  - GIF preview:
    <br>
    <img src="runs/detect/video_quick/preview.gif" alt="Number plate detection video preview" width="520"/>
- Additional demo video (exp10): [runs/detect/exp10/Train yolov7 on the custom dataset.mp4](runs/detect/exp10/Train%20yolov7%20on%20the%20custom%20dataset.mp4)

## Training results (runs/train/yolov7-custom)
The following artifacts are saved during training to help you evaluate and debug the model.

- Confusion matrix
  - Path: `runs/train/yolov7-custom/confusion_matrix.png`
  - Interpretation: For a single class (`number_plate`), the matrix shows the split between correct detections vs false positives/negatives. Off-diagonal pixels indicate confusion with background.
  - Preview:
    <br>
    <img src="runs/train/yolov7-custom/confusion_matrix.png" alt="Confusion matrix" width="520"/>

- Curves (by confidence threshold)
  - Precision curve: `runs/train/yolov7-custom/P_curve.png`
  - Recall curve: `runs/train/yolov7-custom/R_curve.png`
  - F1 curve: `runs/train/yolov7-custom/F1_curve.png`
  - PR curve: `runs/train/yolov7-custom/PR_curve.png`
  - Notes: The F1 peak often suggests a good default `--conf-thres` for inference. The PR curve’s area relates to AP.
  - Preview:
    <br>
    <img src="runs/train/yolov7-custom/P_curve.png" alt="Precision curve" width="320"/>
    <img src="runs/train/yolov7-custom/R_curve.png" alt="Recall curve" width="320"/>
    <br>
    <img src="runs/train/yolov7-custom/F1_curve.png" alt="F1 curve" width="320"/>
    <img src="runs/train/yolov7-custom/PR_curve.png" alt="PR curve" width="320"/>

- Overall training curves
  - Path: `runs/train/yolov7-custom/results.png`
  - Tracks box/class/objectness losses, LR, and metrics (P, R, mAP@.5, mAP@.5:.95) vs epochs.
  - Preview:
    <br>
    <img src="runs/train/yolov7-custom/results.png" alt="Training curves" width="520"/>

- Text summary per epoch
  - Path: `runs/train/yolov7-custom/results.txt`
  - CSV-like lines with timings, losses, metrics.

### Train-batch previews (what is train_batchX.jpg?)
Files like `train_batch0.jpg` … `train_batch9.jpg` are snapshots of augmented training batches. They show images after mosaic/flip/scale/HSV, plus the drawn labels. Use them to spot:
- Label issues (misaligned boxes, wrong class IDs)
- Augmentation problems (over-cropping, extreme scales)
- Class imbalance (few plates per batch)

Example:
<br>
<img src="runs/train/yolov7-custom/train_batch0.jpg" alt="Train batch preview" width="520"/>

Useful files to inspect:
- `runs/train/yolov7-custom/train_batch0.jpg`
- `runs/train/yolov7-custom/train_batch1.jpg`
- `runs/train/yolov7-custom/test_batch0_labels.jpg` vs `runs/train/yolov7-custom/test_batch0_pred.jpg`

## What’s inside
- Custom weights: `best.pt` trained to detect number plates (single class)
- Sample media: `1.jpg`, `1.png`, `1.mp4`
- Inference scripts: `detect.py`, `custom_test_img.py`, `custom_test_video.py`
- Data/configs: `data/custom-data.yaml`, `cfg/training/yolov7-custom.yaml`, hyperparameters in `data/hyp.scratch.*.yaml`
- Core code from YOLOv7 (`models/`, `utils/`, etc.)

## Dataset summary
- Task: Object detection (license/number plates)
- Classes: 1 (e.g., `number_plate`)
- Format: YOLO .txt labels (one file per image)
- Config file: `data/custom-data.yaml` points to your `train/` and `val/` image/label folders

Datasets used:
- Indian Vehicle Number Plate (YOLO annotations): https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
- Car Number Plate Detection: https://www.kaggle.com/datasets/elysian01/car-number-plate-detection

## Environment setup (Windows, PowerShell)
Use a local virtual environment and CPU by default.

```powershell
# from repo root
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# CPU torch/vision (if not already installed)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

GPU users: install a CUDA build of torch/torchvision from https://pytorch.org/get-started/locally/.

## Quick inference
Run detection on the sample image `1.jpg` using your trained weights `best.pt`.
```powershell
python detect.py --weights best.pt --source 1.jpg --img-size 640 --conf-thres 0.25 --device cpu --name quick
```
Outputs go to `runs/detect/<name>/`.

Video/webcam examples:
```powershell
# Video file
python detect.py --weights best.pt --source 1.mp4 --img-size 640 --conf-thres 0.25 --device cpu --name video_demo
# Webcam (0)
python detect.py --weights best.pt --source 0 --img-size 640 --conf-thres 0.25 --device cpu --name cam
```

## Training / fine‑tuning
```powershell
# train from a YOLOv7 base
python train.py --data data/custom-data.yaml --cfg cfg/training/yolov7-custom.yaml --weights yolov7.pt --epochs 100 --img 640

# resume/finetune from custom checkpoint
python train.py --data data/custom-data.yaml --cfg cfg/training/yolov7-custom.yaml --weights best.pt --epochs 50 --img 640
```
Artifacts appear under `runs/train/yolov7-custom/`.

## Evaluation
```powershell
python test.py --data data/custom-data.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device cpu --weights best.pt --name eval
```
Outputs to `runs/test/<name>/` and console.

## Export (optional)
```powershell
python export.py --weights best.pt --img 640 --batch 1 --include onnx
```

## Troubleshooting
- Torch 2.6+ checkpoint loading: This repo includes a small compatibility fix in `models/experimental.py` to force `torch.load(..., weights_only=False)` when loading trusted local checkpoints like `best.pt`.
- Missing torchvision: If you get `ModuleNotFoundError: No module named 'torchvision'`, install `torchvision` to match your installed `torch`.
- Headless: omit `--view-img`; use files under `runs/detect`.

## Acknowledgements & License
- Upstream code: [YOLOv7](https://github.com/WongKinYiu/yolov7) by WongKinYiu et al.
- This repository contains portions of the original code and follows the original license. See `LICENSE.md`.
